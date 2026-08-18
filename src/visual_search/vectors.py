from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .domain import Vector
from .interfaces import VectorIndex


class USearchVectorIndex(VectorIndex):
    def __init__(self, path: Path, dimensions: int, dtype: str = "f32"):
        from usearch.index import Index
        self.path = path
        self.dtype = dtype
        self.index = Index(ndim=dimensions, metric="cos", dtype=dtype)
        if path.exists():
            self.index.load(str(path))

    def add(self, key: int, vector: Vector) -> None:
        if key in self.index:
            self.index.remove(key)
        self.index.add(key, np.asarray(vector, dtype=np.float32))

    def search(self, vector: Vector, limit: int) -> list[tuple[int, float]]:
        if len(self.index) == 0 or not np.any(vector):
            return []
        matches = self.index.search(np.asarray(vector, dtype=np.float32), min(limit, len(self.index)))
        return [(int(key), max(0.0, 1.0 - float(distance))) for key, distance in zip(matches.keys, matches.distances)]

    def remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            if key in self.index:
                self.index.remove(key)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(self.path))

    def contains(self, key: int) -> bool:
        return key in self.index
