from __future__ import annotations
from typing import List
from omegaconf import DictConfig
from ..core.pipeline import PipelineStep, PipelineContext
from ..core.artifacts import Artifact
from .collection import BEIRDatasetCollector, CategoryBasedDatasetCollector
from ..core import paths as pathbuilder
from pathlib import Path

class DataCollectionStep(PipelineStep):
    name = "data_collection"
    
    # Known category-based datasets that don't use BEIR format
    CATEGORY_BASED_DATASETS = {"wos", "bbc-news", "huffpost-news", "mental-health"}

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        # Always run initially; later we can check for existing artifacts
        return True

    def run(self, ctx: PipelineContext) -> List[Artifact]:
        base_out = self.cfg.output_dir
        dataset_name = getattr(self.cfg, "dataset_name", "dataset")
        source = str(getattr(self.cfg, "source", "beir_scifact"))
        
        # Auto-detect category-based datasets if source uses beir_ prefix
        if source.startswith("beir_"):
            dataset_id = source.split("beir_", 1)[1]
            # Check if this is actually a category-based dataset
            if dataset_id in self.CATEGORY_BASED_DATASETS:
                # Redirect to category-based collector
                cfg_dict = dict(self.cfg)
                if not cfg_dict.get("gcs_corpus_path"):
                    cfg_dict["gcs_corpus_path"] = f"gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset_id}/extracted/documents/corpus-and-topics.tsv"
                collector = CategoryBasedDatasetCollector(source_cfg=cfg_dict, dataset_id=dataset_id)
            else:
                # Standard BEIR dataset
                collector = BEIRDatasetCollector(source_cfg=dict(self.cfg), dataset_id=dataset_id)
        elif source.startswith("category_"):
            # Category-based dataset (non-BEIR)
            dataset_id = source.split("category_", 1)[1]
            cfg_dict = dict(self.cfg)
            # Auto-configure GCS path if not explicitly set
            if not cfg_dict.get("gcs_corpus_path"):
                cfg_dict["gcs_corpus_path"] = f"gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset_id}/extracted/documents/corpus-and-topics.tsv"
            collector = CategoryBasedDatasetCollector(source_cfg=cfg_dict, dataset_id=dataset_id)
        else:
            raise ValueError(f"Unsupported data collection source: {source}")
        topics_uri, corpus_uri, qrels_uri = collector.collect_all(base_out, dataset_name)
        merged_path = Path(base_out) / pathbuilder.transformed_documents_and_topics_tsv(dataset_name)

        artifacts = [
            Artifact(name="topics", uri=topics_uri, kind="topics", config_snapshot=dict(self.cfg)),
            Artifact(name="corpus", uri=corpus_uri, kind="corpus", config_snapshot=dict(self.cfg)),
            Artifact(name="qrels", uri=qrels_uri, kind="qrels", config_snapshot=dict(self.cfg)),
            Artifact(
                name="documents_and_topics",
                uri=str(merged_path),
                kind="documents",
                config_snapshot=dict(self.cfg),
            ),
        ]
        return artifacts
