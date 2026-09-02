from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .domain import Vector
from .interfaces import VectorIndex


LOGGER = logging.getLogger(__name__)


def probe_index_file(path: Path, dimensions: int, dtype: str) -> int:
    """Load an index in the current process for the isolated probe command."""
    from usearch.index import Index

    index = Index(ndim=dimensions, metric="cos", dtype=dtype)
    index.load(str(path))
    return len(index)


def _probe_command(path: Path, dimensions: int, dtype: str) -> list[str]:
    arguments = ["--validate-vector-index", str(path), str(dimensions), dtype]
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "visual_search.app", *arguments]


def validate_index_file(path: Path, dimensions: int, dtype: str, timeout: float = 60.0) -> bool:
    """Validate a native USearch file outside the UI process.

    Some malformed USearch files terminate the process with 0xC0000005 instead
    of raising a Python exception.  A short-lived child process contains that
    native failure so the application can rebuild from SQLite safely.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            _probe_command(path, dimensions, dtype),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.exception("Could not validate vector index %s", path)
        return False
    return result.returncode == 0


def _quarantine_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.corrupt-{stamp}-{suffix}{path.suffix}")
        suffix += 1
    return candidate


class USearchVectorIndex(VectorIndex):
    def __init__(self, path: Path, dimensions: int, dtype: str = "f32"):
        from usearch.index import Index
        self.path = path
        self.dtype = dtype
        self.dimensions = dimensions
        self.recovered_from: Path | None = None
        self._lock = threading.RLock()
        self.index = Index(ndim=dimensions, metric="cos", dtype=dtype)
        if path.exists():
            if validate_index_file(path, dimensions, dtype):
                self.index.load(str(path))
            else:
                quarantine = _quarantine_path(path)
                path.replace(quarantine)
                self.recovered_from = quarantine
                LOGGER.error(
                    "Unsafe USearch index was quarantined at %s; rebuilding %s from SQLite",
                    quarantine,
                    path,
                )

    def add(self, key: int, vector: Vector) -> None:
        with self._lock:
            if key in self.index:
                self.index.remove(key)
            self.index.add(key, np.asarray(vector, dtype=np.float32))

    def search(self, vector: Vector, limit: int) -> list[tuple[int, float]]:
        with self._lock:
            if len(self.index) == 0 or not np.any(vector):
                return []
            matches = self.index.search(np.asarray(vector, dtype=np.float32), min(limit, len(self.index)))
            return [(int(key), max(0.0, 1.0 - float(distance))) for key, distance in zip(matches.keys, matches.distances)]

    def remove(self, keys: Iterable[int]) -> None:
        with self._lock:
            for key in keys:
                if key in self.index:
                    self.index.remove(key)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(
                f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                self.index.save(str(temporary))
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def contains(self, key: int) -> bool:
        with self._lock:
            return key in self.index

    def size(self) -> int:
        with self._lock:
            return len(self.index)
