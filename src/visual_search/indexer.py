from __future__ import annotations

import hashlib
import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from .domain import Asset, SearchDebugTrace, SearchHit, SearchMode, SearchSession
from .gpu_scheduler import GpuPriority, GpuRuntimeUnavailable, GpuScheduler
from .interfaces import EmbeddingEngine, ObjectDetector, OcrEngine, VectorIndex
from .scoring import effective_weights, merge_hits
from .storage import SQLiteAssetRepository, asset_matches_metadata
from .text import classify_query, metadata_filters
from .vlm import VlmReranker

LOGGER = logging.getLogger(__name__)
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
METADATA_VERSION = "metadata-thumbnail-v2"
REGION_LAYOUT_VERSION = "adaptive-regions-v2-min1024"


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
    if width <= 0 or height <= 0 or max(width, height) < 1024 or min(width, height) < 768:
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
        self.last_stop_reason = ""
        self._scan_lock = threading.Lock()
        self.repository.queue_stale_stage("metadata", METADATA_VERSION)
        self.repository.queue_stale_stage("ocr", self.ocr_version)
        self.repository.queue_stale_stage("embedding", self.embedding_version)
        if self.region_vectors is not None:
            self.repository.queue_stale_stage("region_embedding", self.region_version)
        if self.objects is not None:
            self.repository.queue_stale_stage("object", self.object_version)
        self.repository.queue_resolved_failures()
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
            required.append(self.repository.stage_complete(
                asset_id, "region_embedding", self.region_version, checksum
            ))
        if self.objects is not None:
            required.append(self.repository.stage_ready(asset_id, "object", self.object_version, checksum))
        return all(required)

    def scan(self, root: Path, progress: Callable[[int, int, str], None] | None = None,
             selected_paths: list[Path] | None = None, cleanup: bool = True) -> None:
        with self._scan_lock:
            self._scan_locked(root, progress, selected_paths, cleanup)

    def _scan_locked(self, root: Path, progress: Callable[[int, int, str], None] | None,
                     selected_paths: list[Path] | None, cleanup: bool) -> None:
        self.cancel_event.clear()
        self.last_stop_reason = ""
        self.repository.recover_interrupted_stages()
        if selected_paths is None:
            if not root.is_dir():
                if progress:
                    progress(0, 0, "資料夾目前無法存取；既有索引已保留")
                return
            files = [item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED]
        else:
            files = [
                Path(item) for item in dict.fromkeys(selected_paths)
                if Path(item).is_file() and Path(item).suffix.lower() in SUPPORTED
            ]
        self.repository.add_root(str(root))
        vectors_changed = False
        completed_positions = 0
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
                    completed_positions = position
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
                    except GpuRuntimeUnavailable as exc:
                        self._pause_after_gpu_fault(
                            asset_id, "ocr", self.ocr_version, checksum, exc, errors
                        )
                        break
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
                    except GpuRuntimeUnavailable as exc:
                        self._pause_after_gpu_fault(
                            asset_id, "object", self.object_version, checksum, exc, errors
                        )
                        break
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
                    except GpuRuntimeUnavailable as exc:
                        self._pause_after_gpu_fault(
                            asset_id, "embedding", self.embedding_version, checksum, exc, errors
                        )
                        break
                    except Exception as exc:
                        message = f"embedding: {exc}"
                        errors.append(message)
                        self.repository.set_stage(asset_id, "embedding", "error", self.embedding_version, checksum, str(exc))
                        LOGGER.exception("Embedding failed for %s", path)

                if self.region_vectors is not None and not self.repository.stage_complete(
                        asset_id, "region_embedding", self.region_version, checksum):
                    self.repository.set_stage(
                        asset_id, "region_embedding", "running", self.region_version, checksum
                    )
                    try:
                        boxes = adaptive_region_boxes(asset.width, asset.height)
                        old_regions = self.repository.region_ids(asset_id)
                        self.region_vectors.remove(old_regions)
                        if not boxes:
                            self.repository.replace_region_embeddings(
                                asset_id, [], [], self.region_version,
                            )
                            vectors_changed = vectors_changed or bool(old_regions)
                            self.repository.set_stage(
                                asset_id, "region_embedding", "skipped", self.region_version, checksum,
                                "Image is below the region-index size threshold",
                            )
                            boxes = []
                            vectors = []
                        else:
                            vectors = self._background_infer(
                                lambda: self.embeddings.image_regions(path, boxes)
                            )
                            if len(vectors) != len(boxes):
                                raise RuntimeError(
                                    f"Region encoder returned {len(vectors)} vectors for {len(boxes)} boxes"
                                )
                        regions = self.repository.replace_region_embeddings(
                            asset_id, boxes, [np.asarray(item, dtype=np.float32) for item in vectors],
                            self.region_version,
                        )
                        for region in regions:
                            if region.id is not None:
                                self.region_vectors.add(region.id, region.vector)
                        vectors_changed = vectors_changed or bool(old_regions or regions)
                        if boxes:
                            self.repository.set_stage(
                                asset_id, "region_embedding", "ready", self.region_version, checksum
                            )
                    except GpuRuntimeUnavailable as exc:
                        self._pause_after_gpu_fault(
                            asset_id, "region_embedding", self.region_version, checksum, exc, errors
                        )
                        break
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
                if all(status in {"ready", "skipped"} for status in required):
                    self.repository.set_asset_status(asset_id, "ready")
                elif any(status == "ready" for status in required):
                    self.repository.set_asset_status(asset_id, "partial", "; ".join(errors))
                else:
                    self.repository.set_asset_status(asset_id, "error", "; ".join(errors) or "All index stages failed")
                completed_positions = position
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
        if cleanup and selected_paths is None:
            self.clean_missing(root)
        if progress:
            if self.last_stop_reason:
                progress(completed_positions, len(files), self.last_stop_reason)
            elif self.cancel_event.is_set():
                progress(completed_positions, len(files), "已取消；下次可接續未完成階段")
            else:
                progress(len(files), len(files), "完成")

    def _pause_after_gpu_fault(
        self, asset_id: int, stage: str, model_version: str, checksum: str,
        error: GpuRuntimeUnavailable, errors: list[str],
    ) -> None:
        summary = str(error).splitlines()[0]
        errors.append(f"{stage}: {summary}")
        self.repository.set_stage(
            asset_id, stage, "pending", model_version, checksum,
            "GPU runtime fault; restart required before retry",
        )
        self.repository.set_asset_status(asset_id, "partial", "; ".join(errors))
        self.last_stop_reason = "GPU 資源錯誤，索引已安全暫停；請重新啟動程式後按「重新掃描」續跑"
        LOGGER.error("Indexing paused after fatal GPU error in %s: %s", stage, error)

    def clean_missing(self, root: Path | None = None) -> None:
        if root is not None and not root.is_dir():
            return
        resolved_root = root.resolve() if root is not None else None
        missing = []
        for asset in self.repository.all_assets():
            if asset.id is None:
                continue
            asset_path = Path(asset.path)
            if resolved_root is not None:
                try:
                    if not asset_path.resolve().is_relative_to(resolved_root):
                        continue
                except (OSError, ValueError):
                    continue
            if not asset_path.exists():
                missing.append(asset.id)
        region_ids = [region_id for asset_id in missing for region_id in self.repository.region_ids(asset_id)]
        self.vectors.remove(missing)
        if self.region_vectors is not None:
            self.region_vectors.remove(region_ids)
        self.repository.delete_ids(missing)
        if missing:
            self.vectors.save()
            if self.region_vectors is not None:
                self.region_vectors.save()

    def remove_path(self, path: Path) -> bool:
        with self._scan_lock:
            asset = self.repository.get_by_path(str(path))
            if asset is None or asset.id is None:
                return False
            region_ids = self.repository.region_ids(asset.id)
            self.vectors.remove([asset.id])
            if self.region_vectors is not None:
                self.region_vectors.remove(region_ids)
            self.repository.delete_ids([asset.id])
            self.vectors.save()
            if self.region_vectors is not None:
                self.region_vectors.save()
            return True

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(root.resolve())
        except (OSError, ValueError):
            return False

    def remove_root(self, root: Path, keep_roots: list[Path] | None = None) -> int:
        """Stop tracking a root and remove only assets no longer covered elsewhere."""
        root = Path(root)
        keep_roots = [Path(item) for item in (keep_roots or [])]
        with self._scan_lock:
            removable: list[int] = []
            thumbnails: list[Path] = []
            for asset in self.repository.all_assets():
                if asset.id is None:
                    continue
                asset_path = Path(asset.path)
                if not self._inside(asset_path, root):
                    continue
                if any(self._inside(asset_path, kept) for kept in keep_roots):
                    continue
                removable.append(int(asset.id))
                if asset.thumbnail:
                    thumbnails.append(Path(asset.thumbnail))
            region_ids = [
                region_id for asset_id in removable
                for region_id in self.repository.region_ids(asset_id)
            ]
            self.vectors.remove(removable)
            if self.region_vectors is not None:
                self.region_vectors.remove(region_ids)
            self.repository.delete_ids(removable)
            self.repository.remove_root(str(root))
            retained_thumbnails = {
                str(Path(asset.thumbnail).resolve()).casefold()
                for asset in self.repository.all_assets() if asset.thumbnail
            }
            for thumbnail in thumbnails:
                try:
                    resolved = thumbnail.resolve()
                    if (
                        resolved.is_relative_to(self.thumbnails.resolve())
                        and str(resolved).casefold() not in retained_thumbnails
                    ):
                        resolved.unlink(missing_ok=True)
                except (OSError, ValueError):
                    LOGGER.warning("Unable to remove orphan thumbnail %s", thumbnail, exc_info=True)
            if removable:
                self.vectors.save()
                if self.region_vectors is not None:
                    self.region_vectors.save()
            return len(removable)


class SearchService:
    def __init__(self, repository: SQLiteAssetRepository, vectors: VectorIndex, embeddings: EmbeddingEngine,
                 scheduler: GpuScheduler | None = None, reranker: VlmReranker | None = None,
                 region_vectors: VectorIndex | None = None):
        self.repository, self.vectors, self.embeddings = repository, vectors, embeddings
        self.scheduler = scheduler
        self.reranker = reranker
        self.region_vectors = region_vectors
        self.last_rerank_source = "deterministic"
        self._debug_enabled = False
        self._debug_lock = threading.Lock()
        self._debug_traces: dict[int, SearchDebugTrace] = {}
        self._sessions: dict[int, SearchSession] = {}

    def set_debug_enabled(self, enabled: bool) -> None:
        self._debug_enabled = bool(enabled)
        if not enabled:
            with self._debug_lock:
                self._debug_traces.clear()

    def debug_trace(self, generation: int) -> SearchDebugTrace | None:
        with self._debug_lock:
            return self._debug_traces.get(int(generation))

    def _store_debug_trace(self, trace: SearchDebugTrace) -> None:
        if not self._debug_enabled:
            return
        with self._debug_lock:
            self._debug_traces[trace.generation] = trace
            for old_generation in sorted(self._debug_traces)[:-8]:
                self._debug_traces.pop(old_generation, None)

    def session(self, generation: int) -> SearchSession | None:
        return self._sessions.get(int(generation))

    def _store_session(self, session: SearchSession) -> None:
        self._sessions[session.generation] = session
        for old_generation in sorted(self._sessions)[:-8]:
            self._sessions.pop(old_generation, None)

    @staticmethod
    def _dynamic_matches(index: VectorIndex, vector: np.ndarray, start: int) -> tuple[list[tuple[int, float]], dict[str, object]]:
        size = max(0, int(getattr(index, "size", lambda: 0)()))
        requested = min(max(1, int(start)), size) if size else max(1, int(start))
        rounds: list[dict[str, object]] = []
        matches: list[tuple[int, float]] = []
        cutoff = 0.05
        stop_reason = "empty"
        while requested > 0:
            matches = index.search(vector, requested)
            if not matches:
                stop_reason = "empty"
                break
            best = float(matches[0][1])
            tail = float(matches[-1][1])
            cutoff = max(0.05, best - 0.25)
            rounds.append({"requested": requested, "returned": len(matches), "best": best,
                           "tail": tail, "cutoff": cutoff})
            if tail < cutoff:
                stop_reason = "tail_below_cutoff"
                break
            if size and requested >= size:
                stop_reason = "index_exhausted"
                break
            if not size or len(matches) < requested:
                stop_reason = "available_results_exhausted"
                break
            requested = min(size, requested * 2)
        selected = [(key, score) for key, score in matches if float(score) >= cutoff]
        return selected, {"rounds": rounds, "cutoff": cutoff, "stop_reason": stop_reason,
                          "selected": len(selected), "index_size": size}

    def search(self, query: str, limit: int | None = None, progress=None,
               generation: int | None = None,
               mode: SearchMode = SearchMode.SMART_COMPLETE) -> list[SearchHit]:
        if generation is None:
            generation = self.scheduler.next_generation() if self.scheduler is not None else 0
        if not isinstance(mode, SearchMode):
            mode = SearchMode(str(mode))
        return self._search(query, limit, progress, generation, mode)

    def _search(self, query: str, limit: int | None, progress, generation: int,
                mode: SearchMode) -> list[SearchHit]:
        filters = metadata_filters(query)
        semantic = []
        retrieval_guard = self.scheduler.interactive() if self.scheduler is not None else nullcontext()
        with retrieval_guard:
            if self.scheduler is None:
                query_vector = self.embeddings.text(query)
            else:
                query_vector = self.scheduler.run(
                    lambda: self.embeddings.text(query), GpuPriority.QUERY_EMBEDDING,
                    generation=generation,
                )
        semantic_matches, semantic_diagnostics = self._dynamic_matches(self.vectors, query_vector, 256)
        for rank, (key, score) in enumerate(semantic_matches, 1):
            asset = self.repository.get(key)
            if (asset and asset.status in ("ready", "partial")
                    and (not filters or asset_matches_metadata(asset, filters))):
                semantic.append(SearchHit(asset, score, semantic_score=score, semantic_rank=rank))
        regions: list[SearchHit] = []
        region_matches: list[tuple[int, float]] = []
        region_diagnostics: dict[str, object] = {"rounds": [], "cutoff": 0.05, "stop_reason": "disabled"}
        if self.region_vectors is not None:
            best: dict[int, tuple[float, Asset, list[int]]] = {}
            region_matches, region_diagnostics = self._dynamic_matches(self.region_vectors, query_vector, 1024)
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
            ordered = sorted(best.values(), key=lambda item: item[0], reverse=True)
            for rank, (score, asset, box) in enumerate(ordered, 1):
                regions.append(SearchHit(
                    asset, score, region_rank=rank, region_score=score, matched_region=box,
                    reasons=["局部語意命中"],
                ))
        textual_raw = self.repository.text_search(query, None)
        textual = textual_raw
        if filters:
            textual = [hit for hit in textual if asset_matches_metadata(hit.asset, filters)]
        metadata = self.repository.metadata_search(query, None)
        objects_raw = self.repository.object_search(query, None)
        objects = objects_raw
        if filters:
            objects = [hit for hit in objects if asset_matches_metadata(hit.asset, filters)]
        results = merge_hits(
            query, semantic, textual,
            None, objects=objects, metadata=metadata, regions=regions,
        )
        if mode == SearchMode.EXHAUSTIVE:
            present = {hit.asset.id for hit in results if hit.asset.id is not None}
            query_type = classify_query(query)
            for asset in self.repository.eligible_assets(filters):
                if asset.id in present:
                    continue
                results.append(SearchHit(
                    asset, 0.0, deterministic_score=0.0, query_type=query_type,
                    reasons=["徹底檢查：整庫候選"],
                ))
        reranker_requested = bool(self.reranker and self.reranker.should_run(query, results))
        trace = None
        if self._debug_enabled:
            query_type = classify_query(query)
            available = {
                name for name, values in (
                    ("semantic", semantic), ("region", regions), ("text", textual),
                    ("object", objects), ("metadata", metadata),
                ) if values
            }
            trace = SearchDebugTrace(
                generation=generation,
                query=query,
                query_type=query_type,
                filters=filters,
                candidate_counts={
                    "semantic_raw": len(semantic_matches),
                    "semantic": len(semantic),
                    "region_vectors_raw": len(region_matches),
                    "region": len(regions),
                    "text_raw": len(textual_raw),
                    "text": len(textual),
                    "object_raw": len(objects_raw),
                    "object": len(objects),
                    "metadata": len(metadata),
                    "union": len(results),
                },
                effective_weights=effective_weights(query_type, available),
                index_coverage=self.repository.index_coverage(),
                deterministic_candidates=list(results),
                reranker_requested=reranker_requested,
                dynamic_retrieval={"semantic": semantic_diagnostics, "region": region_diagnostics,
                                   "mode": mode.value},
            )
            self._store_debug_trace(trace)
        if self.reranker is not None:
            rerank_guard = (
                self.scheduler.interactive()
                if self.scheduler is not None and mode == SearchMode.SMART_COMPLETE and reranker_requested
                else nullcontext()
            )
            with rerank_guard:
                session = self.reranker.rerank_session(query, results, generation, progress, mode)
            self._store_session(session)
            final = session.results
            self.last_rerank_source = (
                final[0].rerank_source if final else session.reranker_source
            )
            if trace is not None:
                trace.final_results = list(final)
                trace.reranker_source = self.last_rerank_source
                trace.candidate_counts["vlm_input"] = len(results) if trace.reranker_requested else 0
                trace.candidate_counts["vlm_evaluated"] = session.reviewed
                trace.candidate_counts["final"] = len(final)
                self._store_debug_trace(trace)
            return final if limit is None else final[:max(0, int(limit))]
        self.last_rerank_source = "deterministic"
        session = SearchSession(generation, query, mode, len(results), reviewed=len(results),
                                qualified=len(results), complete=True, results=list(results))
        self._store_session(session)
        if trace is not None:
            trace.final_results = list(results)
            trace.reranker_source = self.last_rerank_source
            self._store_debug_trace(trace)
        return results if limit is None else results[:max(0, int(limit))]
