from pathlib import Path

import numpy as np
from PIL import Image

from visual_search.domain import Asset, ObjectDetection, SearchHit
from visual_search.indexer import adaptive_region_boxes
from visual_search.scoring import merge_hits
from visual_search.storage import SQLiteAssetRepository
from visual_search.vlm import create_region_review_canvas
from visual_search.yolo_openvino import parse_yolo_output


def make_asset(tmp_path: Path, name: str = "image.png") -> Asset:
    return Asset(None, str(tmp_path / name), 1, 1, "hash", 3840, 2160, "thumb")


def test_adaptive_regions_are_bounded_and_never_exceed_six():
    assert adaptive_region_boxes(900, 600) == []
    assert len(adaptive_region_boxes(1024, 1024)) == 4
    assert len(adaptive_region_boxes(1920, 1080)) == 4
    assert len(adaptive_region_boxes(3840, 2160)) == 6
    ultrawide = adaptive_region_boxes(3840, 1080)
    assert len(ultrawide) == 3
    assert all(0 <= x1 < x2 <= 3840 and 0 <= y1 < y2 <= 1080 for x1, y1, x2, y2 in ultrawide)


def test_region_vectors_round_trip_as_fp16_and_batch_lookup(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(make_asset(tmp_path))
    regions = repository.replace_region_embeddings(
        asset_id,
        [[0, 0, 100, 100], [80, 0, 180, 100]],
        [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32)],
        "region-v1",
    )
    assert repository.region_count() == 2
    loaded = repository.region_embeddings("region-v1", 3)
    assert len(loaded) == 2
    assert loaded[0].vector.dtype == np.float32
    matches = repository.region_lookup_many([int(item.id) for item in regions if item.id is not None])
    assert {tuple(box) for _asset, box in matches.values()} == {(0, 0, 100, 100), (80, 0, 180, 100)}


def test_visual_rrf_uses_region_channel_even_without_object_label(tmp_path):
    asset = make_asset(tmp_path)
    asset.id = 7
    region = SearchHit(asset, 0.8, region_rank=1, region_score=0.8, matched_region=[0, 0, 100, 100])
    result = merge_hits("桌面上的頻譜分析儀", [], [], regions=[region])
    assert result[0].region_rank == 1
    assert result[0].matched_region == [0, 0, 100, 100]
    assert "region" in result[0].channel_scores


def test_region_review_canvas_contains_full_image_and_crop(tmp_path):
    image = tmp_path / "wide.png"
    Image.new("RGB", (1600, 900), "red").save(image)
    review = create_region_review_canvas(image, [0, 0, 400, 400])
    assert review is not None
    try:
        with Image.open(review) as output:
            assert output.size == (896, 896)
    finally:
        review.unlink(missing_ok=True)


def test_yoloe_nms_segmentation_output_ignores_mask_coefficients():
    output = np.zeros((1, 300, 38), dtype=np.float32)
    output[0, 0, :6] = [100, 120, 300, 360, 0.91, 17]
    boxes, scores, classes = parse_yolo_output(output, 0.20, class_count=80)
    assert boxes.shape == (1, 4)
    assert np.allclose(boxes[0], [100, 120, 200, 240])
    assert classes.tolist() == [17]
    assert float(scores[0]) > 0.90


def test_new_fixed_vocabulary_label_is_searchable_without_code_change(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(make_asset(tmp_path, "lab.png"))
    repository.replace_objects(asset_id, [ObjectDetection("spectrum analyzer", 0.88, [0, 0, 9, 9])])
    repository.set_asset_status(asset_id, "ready")
    result = repository.object_search("find a spectrum analyzer on the desk")
    assert result and result[0].asset.id == asset_id
