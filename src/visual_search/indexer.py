from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from .domain import Asset, SearchHit
from .gpu_scheduler import GpuPriority, GpuScheduler
from .interfaces import EmbeddingEngine, ObjectDetector, OcrEngine, VectorIndex
from .scoring import merge_hits
from .storage import SQLiteAssetRepository, asset_matches_metadata
from .text import metadata_filters
from .vlm import VlmReranker

LOGGER = logging.getLogger(__name__)
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
METADATA_VERSION = "metadata-thumbnail-v2"
REGION_LAYOUT_VERSION = "adaptive-regions-v1"


def _axis_starts(length: int, cells: int, overlap: float = 0.15) -> list[tuple[int, int]]:
    if cells <= 1:
        return [(0, length)]
    cell = max(1, round(length / (cells - (cells - 1) * overlap)))
    step = max(1, round(cell * (1.0 - overlap)))
    starts = [min(index * step, max(0, length - cell)) for index in range(cells)]
    return [(start, min(length, start + cell)) for start in dict.fromkeys(starts)]


def adaptive_region_boxes(width: int, height: int) -> list[list[int]]:
    """Return at most six class-independent crops for small-object retrieval."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0 or max(width, height) < 1280 or min(width, height) < 720:
        return []
    ratio = width / height
    if ratio >= 2.0:
        columns, rows = 3, 1
    elif ratio <= 0.5:
        columns, rows = 1, 3
    elif max(width, height) >= 3840 or width * height >= 8_000_000:
        columns, rows = (3, 2) if width >= height else (2, 3)
    else:
        columns, rows = 2, 2
    xs, ys = _axis_starts(width, columns), _axis_starts(height, rows)
    return [[x1, y1, x2, y2] for y1, y2 in ys for x1, x2 in xs][:6]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IndexController:
    def __init__(self, repository: SQLiteAssetRepository, vectors: VectorIndex, ocr: OcrEngine,
                 embeddings: EmbeddingEngine, thumbnails: Path,
                 objects: ObjectDetector | None = None, scheduler: GpuScheduler | None = None,
                 region_vectors: VectorIndex | None = None):
        self.repository = repository
        self.vectors = vectors
        self.ocr = ocr
        self.embeddings = embeddings
        self.thumbnails = thumbnails
        self.objects = objects
        self.scheduler = scheduler
        self.region_vectors = region_vectors
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()
        self.ocr_version = str(getattr(ocr, "model_version", ocr.__class__.__name__))
        self.embedding_version = str(getattr(embeddings, "model_version", embeddings.__class__.__name__))
        self.region_version = f"{self.embedding_version}-{REGION_LAYOUT_VERSION}"
        self.object_version = str(getattr(objects, "model_version", "")) if objects is not None else ""
        self._restore_vectors()

    def _background_infer(self, function):
        if self.scheduler is None:
            return function()
        self.scheduler.wait_for_background_slot()
        return self.scheduler.run(function, GpuPriority.BACKGROUND_INDEX)

    def _restore_vectors(self) -> None:
        changed = False
        for asset_id, vector in self.repository.embeddings(self.embedding_version, self.embeddings.dimensions):
            if not self.vectors.contains(asset_id):
                self.vectors.add(asset_id, vector)
                changed = True
        if changed:
            self.vectors.save()
        if self.region_vectors is not None:
            region_changed = False
            for region in self.repository.region_embeddings(self.region_version, self.embeddings.dimensions):
                if region.id is not None and not self.region_vectors.contains(region.id):
                    self.region_vectors.add(region.id, region.vector)
                    region_changed = True
            if region_changed:
                self.region_vectors.save()

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.set()

    def _all_stages_ready(self, asset_id: int, checksum: str) -> bool:
        required = [
            self.repository.stage_ready(asset_id, "metadata", METADATA_VERSION, checksum),
            self.repository.stage_ready(asset_id, "ocr", self.ocr_version, checksum),
            self.repository.stage_ready(asset_id, "embedding", self.embedding_version, checksum),
        ]
        if self.region_vectors is not None:
            required.append(self.repository.stage_ready(
                asset_id, "region_embedding", self.region_version, checksum
            ))
        if self.objects is not None:
            required.append(self.repository.stage_ready(asset_id, "object", self.object_version, checksum))
        return all(required)

    def scan(self, root: Path, progress: Callable[[int, int, str], None] | None = None) -> None:
        self.cancel_event.clear()
        self.repository.recover_interrupted_stages()
        files = [item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED]
        self.repository.add_root(str(root))
        vectors_changed = False
        for position, path in enumerate(files, 1):
            if self.cancel_event.is_set():
                break
            self.pause_event.wait()
            if progress:
                progress(position - 1, len(files), path.name)
            try:
                stat = path.stat()
                existing = self.repository.get_by_path(str(path))
                if (
                    existing and existing.id is not None and existing.sha256
                    and existing.size == stat.st_size and existing.modified_ns == stat.st_mtime_ns
                    and self._all_stages_ready(existing.id, existing.sha256)
                ):
                    continue

                checksum = file_hash(path)
                if existing is None:
                    moved = self.repository.get_missing_by_hash(checksum)
                    if moved and moved.id is not None:
                        existing = self.repository.relocate(moved.id, str(path), stat.st_size, stat.st_mtime_ns)

                content_changed = bool(existing and existing.sha256 and existing.sha256 != checksum)
                if existing and existing.id is not None and content_changed:
                    old_regions = self.repository.region_ids(existing.id)
                    self.repository.invalidate_stages(existing.id)
                    self.vectors.remove([existing.id])
                    if self.region_vectors is not None:
                        self.region_vectors.remove(old_regions)
                    vectors_changed = True

                asset = existing or Asset(
                    id=None, path=str(path), size=stat.st_size, modified_ns=stat.st_mtime_ns,
                    sha256=checksum, width=0, height=0, thumbnail="",
                )
                asset.path = str(path)
                asset.size = stat.st_size
                asset.modified_ns = stat.st_mtime_ns
                asset.sha256 = checksum
                asset.status = "indexing"
                asset.error = ""
                asset_id = self.repository.upsert(asset)
                asset.id = asset_id
                errors: list[str] = []

                if not self.repository.stage_ready(asset_id, "metadata", METADATA_VERSION, checksum):
                    self.repository.set_stage(asset_id, "metadata", "running", METADATA_VERSION, checksum)
                    try:
                        with Image.open(path) as source:
                            oriented = ImageOps.exif_transpose(source)
                            asset.width, asset.height = oriented.size
                            thumb_name = f"{checksum[:24]}.jpg"
                            thumb_path = self.thumbnails / thumb_name
                            self.thumbnails.mkdir(parents=True, exist_ok=True)
                            thumb = oriented.convert("RGB")
                            thumb.thumbnail((480, 360))
                            thumb.save(thumb_path, "JPEG", quality=82)
                            asset.thumbnail = str(thumb_path)
                        self.repository.upsert(asset)
                        self.repository.set_stage(asset_id, "metadata", "ready", METADATA_VERSION, checksum)
                    except Exception as exc:
                        message = f"metadata: {exc}"
                        errors.append(message)
                        self.repository.set_stage(asset_id, "metadata", "error", METADATA_VERSION, checksum, str(exc))

                if not self.repository.stage_ready(asset_id, "ocr", self.ocr_version, checksum):
                    self.repository.set_stage(asset_id, "ocr", "running", self.ocr_version, checksum)
                    try:
                        lines = self._background_infer(lambda: self.ocr.extract(path))
                        self.repository.replace_ocr(
                            asset_id, lines, getattr(self.ocr, "last_diagnostics", {}),
                        )
                        asset = self.repository.get(asset_id) or asset
                        self.repository.set_stage(asset_id, "ocr", "ready", self.ocr_version, checksum)
                    except Exception as exc:
                        message = f"ocr: {exc}"
                        errors.append(message)
                        self.repository.replace_ocr(asset_id, [])
                        self.repository.set_stage(asset_id, "ocr", "error", self.ocr_version, checksum, str(exc))
                        LOGGER.exception("OCR failed for %s", path)

                if self.objects is not None and not self.repository.stage_ready(
                        asset_id, "object", self.object_version, checksum):
                    self.repository.set_stage(asset_id, "object", "running", self.object_version, checksum)
                    try:
                        detections = self._background_infer(lambda: self.objects.detect(path))
                        self.repository.replace_objects(asset_id, detections)
                        self.repository.set_stage(asset_id, "object", "ready", self.object_version, checksum)
                    except Exception as exc:
                        message = f"object: {exc}"
                        errors.append(message)
                        self.repository.replace_objects(asset_id, [])
                        self.repository.set_stage(asset_id, "object", "error", self.object_version, checksum, str(exc))
                        LOGGER.exception("Object detection failed for %s", path)

                if not self.repository.stage_ready(asset_id, "embedding", self.embedding_version, checksum):
                    self.repository.set_stage(asset_id, "embedding", "running", self.embedding_version, checksum)
                    try:
                        vector = np.asarray(self._background_infer(lambda: self.embeddings.image(path)), dtype=np.float32)
                        self.repository.save_embedding(asset_id, vector, self.embedding_version)
                        self.vectors.add(asset_id, vector)
                        vectors_changed = True
                        asset.model_version = self.embedding_version
                        self.repository.upsert(asset)
                        self.repository.set_stage(asset_id, "embedding", "ready", self.embedding_version, checksum)
                    except Exception as exc:
                        message = f"embedding: {exc}"
                        errors.append(message)
                        self.repository.set_stage(asset_id, "embedding", "error", self.embedding_version, checksum, str(exc))
                        LOGGER.exception("Embedding failed for %s", path)

                if self.region_vectors is not None and not self.repository.stage_ready(
                        asset_id, "region_embedding", self.region_version, checksum):
                    self.repository.set_stage(
                        asset_id, "region_embedding", "running", self.region_version, checksum
                    )
                    try:
                        boxes = adaptive_region_boxes(asset.width, asset.height)
                        vectors = self._background_infer(
                            lambda: self.embeddings.image_regions(path, boxes)
                        ) if boxes else []
                        old_regions = self.repository.region_ids(asset_id)
                        self.region_vectors.remove(old_regions)
                        regions = self.repository.replace_region_embeddings(
                            asset_id, boxes, [np.asarray(item, dtype=np.float32) for item in vectors],
                            self.region_version,
                        )
                        for region in regions:
                            if region.id is not None:
                                self.region_vectors.add(region.id, region.vector)
                        vectors_changed = vectors_changed or bool(old_regions or regions)
                        self.repository.set_stage(
                            asset_id, "region_embedding", "ready", self.region_version, checksum
                        )
                    except Exception as exc:
                        message = f"region_embedding: {exc}"
                        errors.append(message)
                        self.repository.set_stage(
                            asset_id, "region_embedding", "error", self.region_version, checksum, str(exc)
                        )
                        LOGGER.exception("Region embedding failed for %s", path)

                states = self.repository.stage_states(asset_id)
                stage_names = ["metadata", "ocr", "embedding"]
                if self.region_vectors is not None:
                    stage_names.append("region_embedding")
                if self.objects is not None:
                    stage_names.append("object")
                required = [states.get(name, {}).get("status") for name in stage_names]
                if all(status == "ready" for status in required):
                    self.repository.set_asset_status(asset_id, "ready")
                elif any(status == "ready" for status in required):
                    self.repository.set_asset_status(asset_id, "partial", "; ".join(errors))
                else:
                    self.repository.set_asset_status(asset_id, "error", "; ".join(errors) or "All index stages failed")
            except Exception as exc:
                LOGGER.exception("Unable to index %s", path)
                try:
                    stat = path.stat()
                    failed = Asset(
                        id=None, path=str(path), size=stat.st_size, modified_ns=stat.st_mtime_ns,
                        sha256="", width=0, height=0, thumbnail="", status="error", error=str(exc),
                    )
                    self.repository.upsert(failed, [])
                except Exception:
                    LOGGER.exception("Unable to record indexing failure for %s", path)
        if vectors_changed:
            self.vectors.save()
            if self.region_vectors is not None:
                self.region_vectors.save()
        self.clean_missing()
        if progress:
            progress(len(files), len(files), "完成")

    def clean_missing(self) -> None:
        missing = [asset.id for asset in self.repository.all_assets() if asset.id is not None and not Path(asset.path).exists()]
        region_ids = [region_id for asset_id in missing for region_id in self.repository.region_ids(asset_id)]
        self.vectors.remove(missing)
        if self.region_vectors is not None:
            self.region_vectors.remove(region_ids)
        self.repository.delete_ids(missing)
        if missing:
            self.vectors.save()
            if self.region_vectors is not None:
                self.region_vectors.save()


class SearchService:
    def __init__(self, repository: SQLiteAssetRepository, vectors: VectorIndex, embeddings: EmbeddingEngine,
                 scheduler: GpuScheduler | None = None, reranker: VlmReranker | None = None,
                 region_vectors: VectorIndex | None = None):
        self.repository, self.vectors, self.embeddings = repository, vectors, embeddings
        self.scheduler = scheduler
        self.reranker = reranker
        self.region_vectors = region_vectors
        self.last_rerank_source = "deterministic"

    def search(self, query: str, limit: int = 10, progress=None,
               generation: int | None = None) -> list[SearchHit]:
        if generation is None:
            generation = self.scheduler.next_generation() if self.scheduler is not None else 0
        if self.scheduler is not None:
            with self.scheduler.interactive():
                return self._search(query, limit, progress, generation)
        return self._search(query, limit, progress, generation)

    def _search(self, query: str, limit: int, progress, generation: int) -> list[SearchHit]:
        deterministic_limit = max(60, limit, self.reranker.candidate_count if self.reranker else 0)
        filters = metadata_filters(query)
        semantic = []
        if self.scheduler is None:
            query_vector = self.embeddings.text(query)
        else:
            query_vector = self.scheduler.run(
                lambda: self.embeddings.text(query), GpuPriority.QUERY_EMBEDDING,
                generation=generation,
            )
        for rank, (key, score) in enumerate(self.vectors.search(query_vector, max(deterministic_limit, 200)), 1):
            asset = self.repository.get(key)
            if (asset and asset.status in ("ready", "partial")
                    and (not filters or asset_matches_metadata(asset, filters))):
                semantic.append(SearchHit(asset, score, semantic_score=score, semantic_rank=rank))
        regions: list[SearchHit] = []
        if self.region_vectors is not None:
            best: dict[int, tuple[float, Asset, list[int]]] = {}
            region_matches = self.region_vectors.search(query_vector, max(500, deterministic_limit))
            lookup = self.repository.region_lookup_many([key for key, _score in region_matches])
            for key, score in region_matches:
                match = lookup.get(key)
                if match is None:
                    continue
                asset, box = match
                if asset.status not in ("ready", "partial") or (filters and not asset_matches_metadata(asset, filters)):
                    continue
                if asset.id is not None and (asset.id not in best or score > best[asset.id][0]):
                    best[asset.id] = (score, asset, box)
            ordered = sorted(best.values(), key=lambda item: item[0], reverse=True)[:200]
            for rank, (score, asset, box) in enumerate(ordered, 1):
                regions.append(SearchHit(
                    asset, score, region_rank=rank, region_score=score, matched_region=box,
                    reasons=["局部語意命中"],
                ))
        textual = self.repository.text_search(query, max(deterministic_limit, 200))
        if filters:
            textual = [hit for hit in textual if asset_matches_metadata(hit.asset, filters)]
        metadata = self.repository.metadata_search(query, max(deterministic_limit, 200))
        objects = self.repository.object_search(query, max(deterministic_limit, 200))
        if filters:
            objects = [hit for hit in objects if asset_matches_metadata(hit.asset, filters)]
        results = merge_hits(
            query, semantic, textual,
            deterministic_limit, objects=objects, metadata=metadata, regions=regions,
        )
        if self.reranker is not None:
            final = self.reranker.rerank(query, results, generation, progress)
            self.last_rerank_source = final[0].rerank_source if final else "qwen3.5"
            return final[:limit]
        self.last_rerank_source = "deterministic"
        return results[:limit]
