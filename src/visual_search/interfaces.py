from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from .domain import Asset, ObjectDetection, OcrLine, SearchHit, Vector, VlmBatchVerdict, VlmVerdict


class OcrEngine(ABC):
    @abstractmethod
    def extract(self, image: Path) -> list[OcrLine]: ...


class EmbeddingEngine(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def image(self, image: Path) -> Vector: ...

    @abstractmethod
    def text(self, text: str) -> Vector: ...

    def image_regions(self, image: Path, boxes: Sequence[Sequence[int]]) -> list[Vector]:
        """Encode several source-image crops in one backend batch when supported."""
        raise NotImplementedError("Region embedding is not supported by this engine")


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: Path) -> list[ObjectDetection]: ...


class VlmEngine(ABC):
    @abstractmethod
    def review(self, original_request: str, visual_target: str, image: Path,
               relevant_ocr: Sequence[str] = ()) -> VlmVerdict: ...

    def review_batch(self, original_request: str, visual_target: str,
                     images: Sequence[Path], relevant_ocr: Sequence[Sequence[str]] = ()) -> list[VlmBatchVerdict]:
        """Default compatibility path for engines that only support one image."""
        rows = list(relevant_ocr)
        return [
            VlmBatchVerdict(index, self.review(
                original_request, visual_target, image,
                rows[index] if index < len(rows) else (),
            ))
            for index, image in enumerate(images)
        ]

    @abstractmethod
    def runtime_details(self) -> list[dict[str, object]]: ...


class AssetRepository(ABC):
    @abstractmethod
    def upsert(self, asset: Asset, ocr_lines: list[OcrLine] | None = None) -> int: ...

    @abstractmethod
    def get(self, asset_id: int) -> Asset | None: ...

    @abstractmethod
    def get_by_path(self, path: str) -> Asset | None: ...

    @abstractmethod
    def text_search(self, query: str, limit: int = 100) -> list[SearchHit]: ...

    @abstractmethod
    def list_roots(self) -> list[str]: ...

    @abstractmethod
    def add_root(self, path: str) -> None: ...


class VectorIndex(ABC):
    @abstractmethod
    def add(self, key: int, vector: Vector) -> None: ...

    @abstractmethod
    def search(self, vector: Vector, limit: int) -> list[tuple[int, float]]: ...

    @abstractmethod
    def remove(self, keys: Iterable[int]) -> None: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def contains(self, key: int) -> bool: ...

    def size(self) -> int:
        return 0
