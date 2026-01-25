from __future__ import annotations
from typing import List, Dict, Any
import logging
from pathlib import Path
import tempfile
from dataclasses import dataclass

try:
    from pyUDLF import run_calls as udlf
    from pyUDLF.utils import inputType
    PYUDLF_AVAILABLE = True
except ImportError:  # pragma: no cover
    udlf = None
    inputType = None
    PYUDLF_AVAILABLE = False


logger = logging.getLogger("udlf_text_expresso")


@dataclass(frozen=True)
class UDLFInputArtifacts:
    ranked_list_path: Path
    lists_path: Path
    num_queries: int
    num_docs: int


def prepare_udlf_input_files(rows: List[Dict[str, Any]], base_dir: Path) -> UDLFInputArtifacts:
    """Transform retrieval rows into pyUDLF ranked-list and list files."""
    grouped: Dict[str, List[str]] = {}
    doc_pool: Dict[str, None] = {}

    for row in rows:
        qid = str(row.get("query_id"))
        did = str(row.get("doc_id"))
        grouped.setdefault(qid, []).append(did)
        doc_pool.setdefault(did, None)

    if not doc_pool:
        raise ValueError("No document identifiers available for UDLF reranking.")

    doc_tokens = sorted(doc_pool.keys())

    ranked_list_path = base_dir / "ranked-list.txt"
    lists_path = base_dir / "lists.txt"

    with ranked_list_path.open("w", encoding="utf-8") as f:
        for qid, doc_list in grouped.items():
            doc_line = " ".join(doc_list)
            if doc_line:
                f.write(f"{qid} {doc_line}\n")
            else:
                f.write(f"{qid}\n")

    with lists_path.open("w", encoding="utf-8") as f:
        for did in doc_tokens:
            f.write(f"{did}\n")

    return UDLFInputArtifacts(
        ranked_list_path=ranked_list_path,
        lists_path=lists_path,
        num_queries=len(grouped),
        num_docs=len(doc_tokens),
    )


def parse_udlf_output_to_trec(output_file: Path) -> List[Dict[str, Any]]:
    """Parse UDLF ranked-list output and convert to TREC format with linear scores."""
    if not output_file.exists():
        raise FileNotFoundError(f"UDLF output file not found: {output_file}")

    reranked_rows: List[Dict[str, Any]] = []
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            qid = parts[0]
            doc_list = parts[1:]
            # Convert to TREC format: linear scores based on rank (higher rank = higher score)
            total_docs = len(doc_list)
            for rank, did in enumerate(doc_list, start=1):
                # Linear score: starts at total_docs and decreases by 1 for each rank
                rerank_score = float(total_docs - rank + 1)
                reranked_rows.append({
                    "query_id": qid,
                    "doc_id": did,
                    "score": rerank_score,
                    "rank": rank,
                })
    return reranked_rows


class UDLFReranker:
    def __init__(self, method_name: str, config: Dict[str, Any]):
        self.method_name = method_name.upper()
        self.config = config or {}
        
        if not PYUDLF_AVAILABLE:
            logger.warning(
                "pyUDLF package not found; using lightweight score transform for method '%s'",
                method_name,
            )

    def rerank(
        self,
        rows: List[Dict[str, Any]],
        ranked_list_path: Path | None = None,
        lists_path: Path | None = None,
    ) -> List[Dict[str, Any]]:
        if not PYUDLF_AVAILABLE:
            raise RuntimeError(
                f"pyUDLF package is required for method '{self.method_name}' but is not installed. "
                "Install it with: pip install pyUDLF"
            )
        
        return self._rerank_with_pyudlf(rows, ranked_list_path, lists_path)

    def _rerank_with_pyudlf(
        self,
        rows: List[Dict[str, Any]],
        ranked_list_path: Path | None = None,
        lists_path: Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Execute rerank using pyUDLF binary wrapper, mirroring the notebook's logic."""
        # 1. Set up paths and parameters from config
        binary_path = self.config.get("binary_path")
        config_path = self.config.get("config_path")
        output_base = self.config.get("output_path")

        if not binary_path:
            raise ValueError("`binary_path` must be provided in the rerank experiment config.")
        if not config_path:
            raise ValueError("`config_path` must be provided in the rerank experiment config.")
        if not output_base:
            raise ValueError("`output_path` must be provided by the runner for pyUDLF execution.")

        # 2. Set binary path and initialize InputType with config.ini, exactly as in the notebook.
        udlf.setBinaryPath(path=binary_path)
        input_data = inputType.InputType(config_path=config_path)

        # 3. Use provided input files or create temporary ones
        if ranked_list_path and lists_path:
            # Use pre-generated files from the pipeline
            if not ranked_list_path.exists():
                raise FileNotFoundError(f"Ranked list file not found: {ranked_list_path}")
            if not lists_path.exists():
                raise FileNotFoundError(f"Lists file not found: {lists_path}")
            
            logger.info(
                "Using pre-generated UDLF input files: ranked_list=%s, lists=%s",
                ranked_list_path,
                lists_path,
            )
            
            # Count queries and docs from the files for logging
            num_queries = sum(1 for line in ranked_list_path.read_text().strip().split('\n') if line)
            num_docs = sum(1 for line in lists_path.read_text().strip().split('\n') if line)

        else:
            # Create temporary files from rows
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                artifacts = prepare_udlf_input_files(rows, tmp_path)
                
                logger.info(
                    "Created temporary UDLF input files: ranked_list=%s, lists=%s",
                    artifacts.ranked_list_path,
                    artifacts.lists_path,
                )
                
                ranked_list_path = artifacts.ranked_list_path
                lists_path = artifacts.lists_path
                num_queries = artifacts.num_queries
                num_docs = artifacts.num_docs

        # 4. Set all parameters on the input_data object
        output_path = Path(output_base) / "data"
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        input_data.set_method_name(self.method_name)
        input_data.set_input_files(str(Path(ranked_list_path).resolve()))
        input_data.set_lists_file(str(Path(lists_path).resolve()))
        input_data.set_param("OUTPUT_FILE_PATH", str(output_path.resolve()))
        input_data.set_param("SIZE_DATASET", num_docs)
        logger.info(
            "UDLF input counts: queries=%d, docs=%d (config=%s)",
            num_queries,
            num_docs,
            input_data.get_param("SIZE_DATASET"),
        )

        def _require_config(key: str) -> Any:
            if key not in self.config:
                raise ValueError(
                    f"UDLF configuration missing required key '{key}' for method '{self.method_name}'."
                )
            value = self.config[key]
            if value is None:
                raise ValueError(
                    f"UDLF configuration value for '{key}' cannot be None (method '{self.method_name}')."
                )
            return value

        window_size = int(_require_config("window_size"))
        input_data.set_param("PARAM_NONE_L", window_size)

        task_value = str(_require_config("task"))
        input_format = str(_require_config("input_format"))
        rk_format = str(_require_config("input_rk_format"))

        input_data.set_param("UDL_TASK", task_value)
        input_data.set_param("INPUT_FILE_FORMAT", input_format)
        input_data.set_param("INPUT_RK_FORMAT", rk_format)

        if self.method_name == "CPRR":
            param_k = int(_require_config("k"))
            param_t = int(_require_config("t"))
            input_data.set_param("PARAM_CPRR_L", window_size)
            input_data.set_param("PARAM_CPRR_K", param_k)
            input_data.set_param("PARAM_CPRR_T", param_t)
        elif self.method_name == "BFSTREE":
            param_k = int(_require_config("k"))
            correlation_metric = str(_require_config("correlation_metric"))
            input_data.set_param("PARAM_BFSTREE_L", window_size)
            input_data.set_param("PARAM_BFSTREE_K", param_k)
            input_data.set_param("PARAM_BFSTREE_CORRELATION_METRIC", correlation_metric)
        elif self.method_name == "RDPAC":
            l_mult = int(_require_config("l_mult"))
            if l_mult <= 0:
                raise ValueError("UDLF configuration 'l_mult' must be positive for method 'RDPAC'.")
            base_l = max(1, window_size // l_mult)
            k_start = int(_require_config("k_start"))
            k_end = int(_require_config("k_end"))
            k_inc = int(_require_config("k_inc"))
            p_value = float(_require_config("p"))
            pl_value = float(_require_config("pl"))
            input_data.set_param("PARAM_RDPAC_K_START", k_start)
            input_data.set_param("PARAM_RDPAC_K_END", k_end)
            input_data.set_param("PARAM_RDPAC_K_INC", k_inc)
            input_data.set_param("PARAM_RDPAC_L", base_l)
            input_data.set_param("PARAM_RDPAC_L_MULT", l_mult)
            input_data.set_param("PARAM_RDPAC_P", p_value)
            input_data.set_param("PARAM_RDPAC_PL", pl_value)

        # 5. Execute pyUDLF with retry logic (UDLF sometimes fails to write output)
        logger.info("Running pyUDLF method '%s' with %d queries", self.method_name, num_queries)
        logger.info("Expected output file: %s", str(output_path) + ".txt")
        
        output_file = Path(f"{output_path}.txt")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                udlf.run(input_data, get_output=True)
                
                # Check if output was created
                if output_file.exists():
                    logger.info("UDLF successfully created output file on attempt %d", attempt)
                    break
                else:
                    logger.warning("UDLF did not create output file on attempt %d/%d", attempt, max_retries)
                    if attempt < max_retries:
                        logger.info("Waiting 3 seconds before retry...")
                        import time
                        time.sleep(3)
                        logger.info("Retrying UDLF execution...")
                        # Clean up incomplete output directory
                        if output_file.parent.exists():
                            import shutil
                            for item in output_file.parent.iterdir():
                                if item != output_file.parent / "config.json":
                                    if item.is_dir():
                                        shutil.rmtree(item)
                                    else:
                                        item.unlink()
                    else:
                        logger.error("UDLF failed after %d attempts", max_retries)
                        logger.error("Checking parent directory: %s", output_file.parent)
                        if output_file.parent.exists():
                            logger.error("Directory exists, contents: %s", list(output_file.parent.iterdir()))
            except Exception as e:
                logger.error("UDLF execution error on attempt %d: %s", attempt, e)
                if attempt >= max_retries:
                    raise

        # 6. Parse UDLF output and convert back to TREC format
        return parse_udlf_output_to_trec(output_file)
