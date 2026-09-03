import json
from pathlib import Path

from PIL import Image, PngImagePlugin

from visual_search.domain import Asset, SearchHit
from visual_search.evaluation import (
    BenchmarkQuery, build_queries, indexed_dataset_paths, load_benchmark_images, score_hits,
)


def make_prompt_image(path: Path, prompt: str) -> None:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps({
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
    }))
    Image.new("RGB", (16, 16), "white").save(path, pnginfo=metadata)


def make_hit(path: Path, score: float = 0.9) -> SearchHit:
    asset = Asset(None, str(path), 1, 1, "sha", 16, 16, str(path), status="ready")
    return SearchHit(asset, score, deterministic_score=score)


def test_prompt_dataset_builds_objective_queries_in_both_languages(tmp_path):
    make_prompt_image(
        tmp_path / "00000001_00001_.png",
        "a red sports car, on a quiet beach, rear view, soft daylight, "
        "photorealistic product photography, highly detailed",
    )
    make_prompt_image(
        tmp_path / "00000002_00001_.png",
        "a golden retriever, in a dense green forest, front view, soft window light, "
        "editorial photography, highly detailed",
    )

    images = load_benchmark_images(tmp_path)
    queries = build_queries(images, "objective", ("en", "zh"))

    assert [image.image_id for image in images] == [1, 2]
    assert any(query.text == "a red sports car" and query.expected_ids == (1,) for query in queries)
    assert any(query.text == "紅色跑車" and query.expected_ids == (1,) for query in queries)
    assert any(query.text == "car" and query.expected_ids == (1,) for query in queries)
    assert any(query.text == "狗" and query.expected_ids == (2,) for query in queries)
    assert any(query.text == "安靜海灘" and query.expected_ids == (1,) for query in queries)
    assert {query.query_type for query in queries} == {"主體", "同義詞", "場景"}


def test_scoring_counts_results_without_manual_review(tmp_path):
    first = tmp_path / "00000001_00001_.png"
    second = tmp_path / "00000002_00001_.png"
    wrong = tmp_path / "00000003_00001_.png"
    for path in (first, second, wrong):
        make_prompt_image(
            path,
            "a red sports car, on a quiet beach, rear view, soft daylight, "
            "photorealistic product photography, highly detailed",
        )
    images = load_benchmark_images(tmp_path)
    query = BenchmarkQuery("Q001-EN", "基礎", "主體", "en", "car", (1, 2))

    row, details = score_hits(
        query,
        [make_hit(second), make_hit(wrong), make_hit(first), make_hit(first)],
        images,
        top_k=3,
        latency_seconds=0.25,
    )

    assert row["returned_total"] == 3
    assert row["correct_total"] == 2
    assert row["precision_returned"] == 2 / 3
    assert row["recall_at_k"] == 1.0
    assert row["hit_at_1"] == 1
    assert row["mrr_at_k"] == 1.0
    assert [detail["correct"] for detail in details] == [1, 0, 1]


def test_index_coverage_matches_exact_paths(tmp_path):
    image_path = tmp_path / "00000001_00001_.png"
    make_prompt_image(
        image_path,
        "a black cat, inside a warehouse, side profile, overcast natural light, "
        "editorial photography, highly detailed",
    )
    images = load_benchmark_images(tmp_path)

    assert indexed_dataset_paths(images, [str(image_path)]) == (1, [])
    assert indexed_dataset_paths(images, []) == (0, [image_path.name])


def test_public_test100_has_exact_objective_query_counts():
    dataset = Path(__file__).resolve().parents[1] / "benchmarks" / "test100" / "images"
    images = load_benchmark_images(dataset)
    objective = build_queries(images, "objective", ("en",))
    mixed = build_queries(images, "objective", ("mixed",))
    chinese = build_queries(images, "objective", ("zh",))
    bilingual = build_queries(images, "objective", ("en", "zh"))

    assert len(images) == 100
    assert len(objective) == 41
    assert len(mixed) == 41
    assert len(chinese) == 41
    assert len(bilingual) == 82
    assert {query.query_type for query in bilingual} == {"主體", "同義詞", "場景"}
    assert {query.language for query in mixed} == {"en", "zh"}
    assert build_queries(images, "core", ("en",)) == objective
    assert build_queries(images, "full", ("en",)) == objective
