import json
from pathlib import Path

from PIL import Image

from visual_search.domain import Asset, SearchHit, VlmVerdict
from visual_search.gpu_scheduler import GpuScheduler
from visual_search.storage import SQLiteAssetRepository
from visual_search.vlm import VlmReranker, build_review_prompt, parse_verdict, visual_review_target


class FakeVlm:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.prompts = []

    def review(self, original_request, visual_target, image, relevant_ocr=()):
        self.prompts.append((original_request, visual_target, Path(image), list(relevant_ocr)))
        return next(self.verdicts)

    def runtime_details(self):
        return []


def test_vlm_prompt_keeps_raw_request_and_excludes_pipeline_hints():
    raw = "找 2026-08 的穿紅衣牽狗的人"
    target = visual_review_target(raw)
    prompt = build_review_prompt(raw, target, ["訂單 123"])
    assert raw in prompt
    assert "穿紅衣牽狗的人" in target
    assert "SigLIP" not in prompt
    assert "YOLO" not in prompt
    assert "RRF" not in prompt


def test_vlm_verdict_parser_is_strict():
    assert parse_verdict('{"match":91,"hard_fail":false}') == VlmVerdict(91, False)
    for invalid in (
        'result: {"match":91,"hard_fail":false}',
        '{"match":91,"hard_fail":false,"reason":"dog"}',
        '{"match":91.5,"hard_fail":false}',
    ):
        try:
            parse_verdict(invalid)
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError("invalid VLM JSON was accepted")


def test_vlm_reranker_filters_hard_fail_but_protects_exact_hits(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    paths = []
    hits = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32, 32), "red").save(path)
        asset = Asset(index + 1, str(path), path.stat().st_size, 1, str(index), 32, 32, str(path))
        paths.append(path)
        hit = SearchHit(asset, 0.8 - index * 0.1, deterministic_score=0.8 - index * 0.1)
        hits.append(hit)
    hits[1].protected_exact_match = True
    engine = FakeVlm([VlmVerdict(10, True), VlmVerdict(95, True)])
    scheduler = GpuScheduler("test-rerank")
    generation = scheduler.next_generation()
    reranker = VlmReranker(repository, engine, scheduler, "always", 2, 5)
    result = reranker.rerank("找紅色圖片", hits, generation)
    scheduler.shutdown()
    assert [item.asset.id for item in result] == [2]
    assert result[0].vlm_hard_fail is False
    assert result[0].rerank_source == "qwen3.5"
    activity = reranker.execution_status()
    assert activity["state"] == "已執行"
    assert activity["candidate_count"] == 2
    assert activity["model_reviewed"] == 2
    assert activity["batch_count"] == 1


def test_vlm_activity_explains_when_exact_query_skips_inference(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "blue").save(path)
    hit = SearchHit(
        Asset(1, str(path), path.stat().st_size, 1, "sha", 32, 32, str(path)),
        0.8, deterministic_score=0.8,
    )
    engine = FakeVlm([VlmVerdict(100, False)])
    scheduler = GpuScheduler("test-skip-rerank")
    reranker = VlmReranker(repository, engine, scheduler, "auto", 0, 5)
    result = reranker.rerank("UX3405", [hit], scheduler.next_generation())
    scheduler.shutdown()

    assert result == [hit]
    activity = reranker.execution_status()
    assert activity["state"] == "本次未執行"
    assert activity["model_reviewed"] == 0
    assert "精確文字" in activity["detail"]
