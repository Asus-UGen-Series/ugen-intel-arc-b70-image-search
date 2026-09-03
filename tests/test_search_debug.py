import numpy as np

from visual_search.domain import Asset
from visual_search.indexer import SearchService
from visual_search.storage import SQLiteAssetRepository


class FakeEmbedding:
    dimensions = 4

    def text(self, _query):
        return np.asarray([1, 0, 0, 0], dtype=np.float32)


class FakeVectors:
    def __init__(self, matches):
        self.matches = matches

    def search(self, _vector, limit):
        return self.matches[:limit]


def test_developer_trace_captures_channel_counts_weights_and_final_results(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(
        None, str(tmp_path / "red-laptop.jpg"), 1, 1, "abc", 32, 32, "thumb.jpg",
        ocr_text="red laptop",
    ))
    service = SearchService(repository, FakeVectors([(asset_id, 0.91)]), FakeEmbedding())
    service.set_debug_enabled(True)

    results = service.search("red laptop", generation=7)
    trace = service.debug_trace(7)

    assert results[0].asset.id == asset_id
    assert trace is not None
    assert trace.query_type == "visual"
    assert trace.candidate_counts["semantic_raw"] == 1
    assert trace.candidate_counts["semantic"] == 1
    assert trace.candidate_counts["text"] == 1
    assert trace.candidate_counts["union"] == 1
    assert trace.final_results[0].asset.id == asset_id
    assert set(trace.effective_weights) == {"semantic", "text"}
    assert abs(sum(trace.effective_weights.values()) - 1.0) < 1e-9


def test_search_trace_is_disabled_by_default(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    service = SearchService(repository, FakeVectors([]), FakeEmbedding())
    service.search("anything", generation=11)
    assert service.debug_trace(11) is None


def test_index_coverage_aggregates_stage_error_reasons(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(
        None, str(tmp_path / "failed.png"), 1, 1, "xyz", 32, 32, "thumb.jpg", status="partial",
    ))
    repository.set_stage(asset_id, "embedding", "error", "v1", "xyz", "CL_OUT_OF_RESOURCES")
    coverage = repository.index_coverage()
    assert coverage["stages"]["embedding"]["error"] == 1
    assert coverage["stage_errors"] == [
        {"stage": "embedding", "error": "CL_OUT_OF_RESOURCES", "count": 1}
    ]
