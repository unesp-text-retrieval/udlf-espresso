from __future__ import annotations
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
from .artifacts import Artifact
from .storage import StorageBackend

logger = logging.getLogger("udlf_text_expresso")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class PipelineContext:
    def __init__(self, run_id: Optional[str], storage: StorageBackend, force: bool = False):
        self.run_id = run_id or datetime.datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
        self.storage = storage
        self.force = force
        self.artifacts: Dict[str, Artifact] = {}

    def register(self, artifact: Artifact) -> None:
        self.artifacts[artifact.name] = artifact
        try:
            local_path = Path(artifact.uri)
        except Exception:
            local_path = None
        if local_path is not None and local_path.exists() and local_path.is_file():
            try:
                self.storage.mirror_local_path(artifact.uri, local_path)
            except NotImplementedError:
                pass

    def get(self, name: str) -> Artifact:
        return self.artifacts[name]


class PipelineStep:
    name: str = "step"

    def should_run(self, ctx: PipelineContext) -> bool:
        return True

    def run(self, ctx: PipelineContext) -> List[Artifact]:  # pragma: no cover - abstract
        raise NotImplementedError

    def execute(self, ctx: PipelineContext) -> List[Artifact]:
        if not ctx.force and not self.should_run(ctx):
            logger.info(f"Skipping step {self.name} (cached)")
            return []
        logger.info(f"Running step {self.name}")
        artifacts = self.run(ctx)
        for a in artifacts:
            ctx.register(a)
        return artifacts


class LinearPipeline:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, ctx: PipelineContext) -> Dict[str, Artifact]:
        for step in self.steps:
            step.execute(ctx)
        return ctx.artifacts
