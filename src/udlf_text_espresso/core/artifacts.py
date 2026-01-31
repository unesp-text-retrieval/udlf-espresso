from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import OmegaConf


@dataclass
class Artifact:
    name: str
    uri: str  # could be local path or s3 uri
    kind: str  # e.g. 'topics', 'corpus', 'retrieval', 'rerank', 'metrics'
    metadata: Dict[str, Any] = field(default_factory=dict)
    config_snapshot: Optional[Dict[str, Any]] = None
    input_artifact_ids: Optional[Dict[str, str]] = None  # name->hash

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update((self.name or "").encode())
        h.update((self.uri or "").encode())
        h.update((self.kind or "").encode())
        if self.config_snapshot:
            try:
                container = OmegaConf.to_container(self.config_snapshot, resolve=True)
            except Exception:
                container = self.config_snapshot
            h.update(json.dumps(container, sort_keys=True, default=str).encode())
        if self.input_artifact_ids:
            h.update(json.dumps(self.input_artifact_ids, sort_keys=True).encode())
        return h.hexdigest()

    def sidecar_path(self) -> str:
        if self.uri.startswith("s3://"):
            return self.uri + ".meta.json"
        p = Path(self.uri)
        return str(p.parent / f"{p.name}.meta.json")
