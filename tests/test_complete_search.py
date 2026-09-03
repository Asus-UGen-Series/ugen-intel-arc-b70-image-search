from pathlib import Path

import numpy as np
from PIL import Image

from visual_search.domain import (
    Asset, ObjectDetection, SearchHit, SearchMode, VlmBatchVerdict, VlmVerdict,
)
from visual_search.gpu_scheduler import GpuScheduler
from visual_search.indexer import SearchService
from visual_search.storage import SQLiteAssetRepository
from visual_search.vlm import VlmReranker, parse_batch_verdicts


class FakeEmbedding:
    dimensions = 2

    def text(self, _query):
        return np.array([1.0, 0.0], dtype=np.float32)


class ExpandingVectors:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def size(self):
        return len(self.rows)

    def search(self, _vector, limit):
        self.calls.append(limit)
        return self.rows[:limit]


class EmptyVectors(ExpandingVectors):
    def __init__(self):
        super().__init__([])


class BatchVlm:
    model_revision = "batch-v1"
    image_size = 448
    batch_size = 4

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0
        self.position = 0

    def review_batch(self, _request, _target, images, _ocr=()):
        self.calls += 1
        result = []
        for candidate_id, _image in enumerate(images):
            score = self.scores[self.position % len(self.scores)]
            self.position += 1
            result.append(VlmBatchVerdict(candidate_id, VlmVerdict(score, False)))
        return result

    def review(self, *_args, **_kwargs):
        score = self.scores[self.position % len(self.scores)]
        self.position += 1
        return VlmVerdict(score, False)

    def runtime_details(self):
        return []


def make_assets(repository, tmp_path: Path, count: int):
    ids = []
    for index in range(count):
        path = tmp_path / f"image-{index:04}.png"
        Image.new("RGB", (8, 8), "red").save(path)
        ids.append(repository.upsert(Asset(
            None, str(path), path.stat().st_size, path.stat().st_mtime_ns,
            f"sha-{index}", 8, 8, str(path), status="ready",
        )))
    return ids


def test_dynamic_vector_retrieval_expands_beyond_200(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    ids = make_assets(repository, tmp_path, 300)
    rows = [(asset_id, 0.90 if index == 0 else 0.80) for index, asset_id in enumerate(ids)]
    rows.extend((10_000 + index, 0.50) for index in range(300))
    vectors = ExpandingVectors(rows)
    service = SearchService(repository, vectors, FakeEmbedding())

    results = service.search("red measuring tool", generation=1)

    assert len(results) == 300
    assert vectors.calls == [256, 512]


def test_exhaustive_mode_includes_every_eligible_asset(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    make_assets(repository, tmp_path, 75)
    service = SearchService(repository, EmptyVectors(), FakeEmbedding())

    results = service.search("find calipers", generation=2, mode=SearchMode.EXHAUSTIVE)

    assert len(results) == 75


def test_vlm_reviews_all_candidates_applies_threshold_and_reuses_cache(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    ids = make_assets(repository, tmp_path, 100)
    hits = [SearchHit(repository.get(asset_id), 0.5, deterministic_score=0.5) for asset_id in ids]
    engine = BatchVlm([75])
    scheduler = GpuScheduler("complete-search")
    generation = scheduler.next_generation()
    reranker = VlmReranker(repository, engine, scheduler, "always", timeout_seconds=5, match_threshold=70)

    first = reranker.rerank_session("find calipers", hits, generation)
    assert first.complete is True
    assert len(first.results) == 100
    assert engine.calls == 25

    cached_engine = BatchVlm([10])
    cached = VlmReranker(repository, cached_engine, scheduler, "always", timeout_seconds=5, match_threshold=70)
    second = cached.rerank_session("find calipers", hits, generation)
    scheduler.shutdown()

    assert second.cached == 100
    assert len(second.results) == 100
    assert cached_engine.calls == 0


def test_vlm_balanced_threshold_rejects_scores_below_70(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    ids = make_assets(repository, tmp_path, 8)
    hits = [SearchHit(repository.get(asset_id), 0.5, deterministic_score=0.5) for asset_id in ids]
    engine = BatchVlm([65, 75])
    scheduler = GpuScheduler("threshold-search")
    generation = scheduler.next_generation()
    session = VlmReranker(
        repository, engine, scheduler, "always", timeout_seconds=5, match_threshold=70,
    ).rerank_session("find calipers", hits, generation)
    scheduler.shutdown()

    assert session.reviewed == 8
    assert session.qualified == 4
    assert session.rejected == 4


def test_batch_verdict_parser_requires_complete_candidate_ids():
    rows = parse_batch_verdicts(
        '{"results":[{"candidate_id":0,"match":91,"hard_fail":false},'
        '{"candidate_id":1,"match":72,"hard_fail":false}]}', 2,
    )
    assert [row.verdict.match for row in rows] == [91, 72]


def test_yoloe_stats_distinguish_processed_zero_and_detected_assets(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    ids = make_assets(repository, tmp_path, 3)
    for asset_id in ids:
        repository.set_stage(asset_id, "object", "ready", "yoloe", f"sha-{asset_id - 1}")
    repository.replace_objects(ids[0], [ObjectDetection("caliper", 0.9, [0, 0, 4, 4])])

    stats = repository.index_coverage()["object_stats"]

    assert stats["ready"] == 3
    assert stats["with_detections"] == 1
    assert stats["zero_detections"] == 2
    assert stats["detections"] == 1
