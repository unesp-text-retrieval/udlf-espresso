from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Set

try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except ImportError:  # pragma: no cover
    gcs_storage = None


class StorageBackend:
    def write_bytes(self, uri: str, data: bytes) -> None:
        raise NotImplementedError

    def read_bytes(self, uri: str) -> bytes:
        raise NotImplementedError

    def exists(self, uri: str) -> bool:
        raise NotImplementedError

    def mirror_local_path(self, uri: str, local_path: Path) -> None:
        if not local_path.exists() or not local_path.is_file():
            return
        with local_path.open("rb") as handle:
            self.write_bytes(uri, handle.read())

    def list_children(self, uri: str) -> List[str]:
        raise NotImplementedError

    def write_json(self, uri: str, obj: Dict[str, Any]) -> None:
        self.write_bytes(uri, json.dumps(obj, indent=2, sort_keys=True).encode())

    def read_json(self, uri: str) -> Dict[str, Any]:
        return json.loads(self.read_bytes(uri))


class LocalStorage(StorageBackend):
    def _path(self, uri: str) -> Path:
        return Path(uri)

    def write_bytes(self, uri: str, data: bytes) -> None:
        p = self._path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)

    def read_bytes(self, uri: str) -> bytes:
        with open(self._path(uri), "rb") as f:
            return f.read()

    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()

    def mirror_local_path(self, uri: str, local_path: Path) -> None:  # pragma: no cover - trivial
        # Local storage already references the same filesystem path; nothing to mirror.
        return

    def list_children(self, uri: str) -> List[str]:
        path = self._path(uri)
        if not path.exists() or not path.is_dir():
            return []
        return sorted([child.name for child in path.iterdir() if child.is_dir()])


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        if boto3 is None:
            raise RuntimeError("boto3 not installed; cannot use S3Storage")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3")

    def _key(self, uri: str) -> str:
        # uri expected like s3://bucket/path/to/file
        if uri.startswith("s3://"):
            parts = uri[5:].split("/", 1)
            key = parts[1] if len(parts) > 1 else ""
        else:
            key = uri
        if self.prefix:
            return f"{self.prefix}/{key}" if not key.startswith(self.prefix) else key
        return key

    def write_bytes(self, uri: str, data: bytes) -> None:
        key = self._key(uri)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read_bytes(self, uri: str) -> bytes:
        key = self._key(uri)
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def exists(self, uri: str) -> bool:
        key = self._key(uri)
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def mirror_local_path(self, uri: str, local_path: Path) -> None:
        if not local_path.exists() or not local_path.is_file():
            return
        key = self._key(uri)
        self._s3.upload_file(str(local_path), self.bucket, key)

    def list_children(self, uri: str) -> List[str]:
        key = self._key(uri)
        if key and not key.endswith("/"):
            key = f"{key}/"
        paginator = self._s3.get_paginator("list_objects_v2")
        seen: Set[str] = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=key, Delimiter="/"):
            for common in page.get("CommonPrefixes", []):
                prefix = common.get("Prefix", "")
                if key and prefix.startswith(key):
                    remainder = prefix[len(key):]
                else:
                    remainder = prefix
                name = remainder.strip("/")
                if name:
                    seen.add(name.split("/", 1)[0])
        return sorted(seen)


class GCSStorage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = "", client: Optional["gcs_storage.Client"] = None):
        if gcs_storage is None:
            raise RuntimeError("google-cloud-storage not installed; cannot use GCSStorage")
        if not bucket:
            raise ValueError("GCSStorage requires a bucket name")
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._client = client or gcs_storage.Client()
        self._bucket = self._client.bucket(self.bucket_name)

    def _blob_name(self, uri: str) -> str:
        if uri.startswith("gs://"):
            parts = uri[5:].split("/", 1)
            key = parts[1] if len(parts) > 1 else ""
        else:
            key = uri
        if self.prefix:
            return f"{self.prefix}/{key}" if not key.startswith(self.prefix) else key
        return key

    def write_bytes(self, uri: str, data: bytes) -> None:
        blob = self._bucket.blob(self._blob_name(uri))
        blob.upload_from_string(data)

    def read_bytes(self, uri: str) -> bytes:
        import logging
        import io
        import time
        logger = logging.getLogger("udlf_text_expresso")
        blob = self._bucket.blob(self._blob_name(uri))
        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: {uri}")
        blob.reload()  # Get metadata including size
        size_mb = blob.size / (1024 * 1024) if blob.size else 0
        logger.info(f"Downloading {size_mb:.2f} MB from GCS: gs://{self.bucket_name}/{self._blob_name(uri)}")
        
        # Use chunked download with progress tracking for large files
        if blob.size and blob.size > 10 * 1024 * 1024:  # > 10 MB
            bytes_io = io.BytesIO()
            chunk_size = 5 * 1024 * 1024  # 5 MB chunks
            start_time = time.time()
            last_log_time = start_time
            
            blob.chunk_size = chunk_size
            blob.download_to_file(bytes_io, timeout=600)  # 10 minute timeout
            
            data = bytes_io.getvalue()
            elapsed = time.time() - start_time
            logger.info(f"Download complete: {len(data) / (1024 * 1024):.2f} MB received in {elapsed:.1f}s ({len(data) / elapsed / (1024 * 1024):.2f} MB/s)")
            return data
        else:
            data = blob.download_as_bytes(timeout=120)
            logger.info(f"Download complete: {len(data)} bytes received")
            return data

    def exists(self, uri: str) -> bool:
        blob = self._bucket.blob(self._blob_name(uri))
        return blob.exists()

    def mirror_local_path(self, uri: str, local_path: Path) -> None:
        if not local_path.exists() or not local_path.is_file():
            return
        blob = self._bucket.blob(self._blob_name(uri))
        blob.upload_from_filename(str(local_path))

    def list_children(self, uri: str) -> List[str]:
        prefix = self._blob_name(uri)
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        iterator = self._client.list_blobs(self.bucket_name, prefix=prefix, delimiter="/")
        seen: Set[str] = set()
        for page in iterator.pages:  # type: ignore[attr-defined]
            for child_prefix in getattr(page, "prefixes", []):
                if prefix and child_prefix.startswith(prefix):
                    remainder = child_prefix[len(prefix):]
                else:
                    remainder = child_prefix
                name = remainder.strip("/")
                if name:
                    seen.add(name.split("/", 1)[0])
        return sorted(seen)
