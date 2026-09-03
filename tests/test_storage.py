from visual_search.domain import Asset, OcrLine
from visual_search.storage import SQLiteAssetRepository


def test_upsert_and_search(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset = Asset(None, str(tmp_path / "ASUS-UX3405.jpg"), 12, 1, "abc", 100, 100, "thumb.jpg", "紅色 筆電")
    asset_id = repository.upsert(asset)
    assert repository.get(asset_id).ocr_text == "紅色 筆電"
    assert repository.text_search("UX3405")[0].asset.id == asset_id


def test_raw_ocr_is_preserved_but_only_searchable_lines_enter_fts(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset = Asset(None, str(tmp_path / "screen.png"), 12, 1, "abc", 100, 100, "thumb.jpg")
    asset_id = repository.upsert(asset)
    repository.replace_ocr(asset_id, [
        OcrLine("這是紅色筆電", 0.93, [[0, 0], [10, 0], [10, 10], [0, 10]], True),
        OcrLine("模糊字串", 0.31, [[0, 20], [10, 20], [10, 30], [0, 30]], False),
    ])
    repository.set_asset_status(asset_id, "ready")
    stored = repository.get(asset_id)
    assert "模糊字串" in stored.ocr_raw_text
    assert "模糊字串" not in stored.ocr_text
    assert repository.text_search("色筆")[0].asset.id == asset_id
    assert repository.text_search("模糊") == []
    assert len(repository.ocr_lines(asset_id)) == 2


def test_pipeline_stage_version_and_embedding_roundtrip(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(None, str(tmp_path / "a.jpg"), 1, 1, "hash", 1, 1, "thumb"))
    repository.set_stage(asset_id, "ocr", "running", "v1", "hash")
    repository.recover_interrupted_stages()
    assert repository.stage_states(asset_id)["ocr"]["status"] == "pending"
    repository.set_stage(asset_id, "ocr", "ready", "v1", "hash")
    assert repository.stage_ready(asset_id, "ocr", "v1", "hash")
    assert not repository.stage_ready(asset_id, "ocr", "v2", "hash")
    repository.save_embedding(asset_id, [0.25, 0.5], "embed-v1")
    restored = repository.embeddings("embed-v1", 2)[0][1]
    assert restored.tolist() == [0.25, 0.5]


def test_code_identifier_search_survives_missing_separator(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(None, str(tmp_path / "code.png"), 1, 1, "hash", 1, 1, "thumb"))
    repository.replace_ocr(asset_id, [OcrLine("minimumconfidence", 0.9, searchable=True)])
    repository.set_asset_status(asset_id, "ready")
    hit = repository.text_search("minimum_confidence")[0]
    assert hit.asset.id == asset_id
    assert any("程式碼識別字匹配" in reason for reason in hit.reasons)


def test_schema_upgrade_only_invalidates_ocr_stage(tmp_path):
    path = tmp_path / "metadata.db"
    repository = SQLiteAssetRepository(path)
    asset_id = repository.upsert(Asset(None, str(tmp_path / "a.png"), 1, 1, "hash", 1, 1, "thumb"))
    repository.set_stage(asset_id, "ocr", "ready", "v2", "hash")
    repository.set_stage(asset_id, "embedding", "ready", "embed", "hash")
    repository._connection().execute("UPDATE app_meta SET value='2' WHERE key='schema_version'")
    repository._connection().commit()
    upgraded = SQLiteAssetRepository(path)
    states = upgraded.stage_states(asset_id)
    assert states["ocr"]["status"] == "pending"
    assert states["embedding"]["status"] == "ready"


def test_explicit_format_metadata_is_a_separate_candidate_channel(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    png_id = repository.upsert(Asset(None, str(tmp_path / "screen.png"), 1, 1, "a", 1, 1, "thumb"))
    repository.upsert(Asset(None, str(tmp_path / "photo.jpg"), 1, 1, "b", 1, 1, "thumb"))
    repository.set_asset_status(png_id, "ready")
    hits = repository.metadata_search("png")
    assert [hit.asset.id for hit in hits] == [png_id]
    assert hits[0].metadata_rank == 1


def test_stale_stage_version_is_queued_and_ready_asset_becomes_partial(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(
        None, str(tmp_path / "image.png"), 1, 1, "hash", 10, 10, "thumb", status="ready",
    ))
    repository.set_stage(asset_id, "region_embedding", "ready", "regions-v1", "hash")
    assert repository.queue_stale_stage("region_embedding", "regions-v2") == 1
    assert repository.stage_states(asset_id)["region_embedding"]["status"] == "pending"
    assert repository.get(asset_id).status == "partial"


def test_resolved_gpu_and_json_failures_are_queued_for_retry(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    first = repository.upsert(Asset(None, str(tmp_path / "a.png"), 1, 1, "a", 10, 10, "thumb"))
    second = repository.upsert(Asset(None, str(tmp_path / "b.png"), 1, 1, "b", 10, 10, "thumb"))
    repository.set_stage(first, "ocr", "error", "v1", "a", "[GPU] CL_OUT_OF_RESOURCES")
    repository.set_stage(second, "object", "error", "v1", "b", "float32 is not JSON serializable")
    assert repository.queue_resolved_failures() == 2
    assert repository.stage_states(first)["ocr"]["status"] == "pending"
    assert repository.stage_states(second)["object"]["status"] == "pending"
