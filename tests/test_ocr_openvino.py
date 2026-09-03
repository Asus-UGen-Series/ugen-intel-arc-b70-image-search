import numpy as np

from PIL import Image

from visual_search.ocr_openvino import (
    DetectedTextBox,
    OpenVinoPaddleOcrEngine,
    decode_ctc,
    decode_image,
    deduplicate_detections,
    preprocess_detection,
    preprocess_recognition,
    recognition_batch_capacity,
    recognition_width,
    should_use_tiles,
    sort_boxes,
    tile_regions,
)


def test_detection_preprocess_uses_stable_aspect_bucket():
    image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    tensor, scale, original = preprocess_detection(image, ((320, 960), (640, 960), (960, 960)))
    assert tensor.shape == (1, 3, 640, 960)
    assert original == (1080, 1920)
    assert scale > 0
    assert np.isfinite(tensor).all()


def test_recognition_preprocess_batches_and_buckets_width():
    crops = [np.zeros((48, 120, 3), dtype=np.uint8), np.zeros((40, 500, 3), dtype=np.uint8)]
    tensor = preprocess_recognition(crops, (160, 320, 640, 1280, 3200))
    assert tensor.shape == (2, 3, 48, 640)


def test_recognition_buckets_use_fixed_memory_budget():
    narrow = np.zeros((48, 250, 3), dtype=np.uint8)
    wide = np.zeros((48, 2500, 3), dtype=np.uint8)
    buckets = (160, 320, 640, 1280, 3200)
    assert recognition_width(narrow, buckets) == 320
    assert recognition_width(wide, buckets) == 3200
    assert recognition_batch_capacity(320, 8) == 8
    assert recognition_batch_capacity(640, 8) == 4
    assert recognition_batch_capacity(1280, 8) == 2
    assert recognition_batch_capacity(3200, 8) == 1


def test_ctc_decode_removes_blank_and_duplicates():
    logits = np.zeros((1, 6, 4), dtype=np.float32)
    for position, token in enumerate((0, 1, 1, 0, 2, 3)):
        logits[0, position, token] = 0.9
    assert decode_ctc(logits, ["blank", "A", "B", "C"])[0][0] == "ABC"


def test_box_sorting_keeps_reading_order():
    right = np.asarray([[100, 10], [140, 10], [140, 30], [100, 30]], dtype=np.float32)
    left = np.asarray([[10, 12], [50, 12], [50, 32], [10, 32]], dtype=np.float32)
    assert np.array_equal(sort_boxes([right, left])[0], left)


def test_unicode_windows_style_path_can_be_decoded(tmp_path):
    path = tmp_path / "螢幕擷取畫面 測試.png"
    Image.new("RGB", (40, 30), "red").save(path)
    decoded = decode_image(path)
    assert decoded.shape == (30, 40, 3)


def test_low_confidence_ocr_is_preserved_for_diagnostics():
    engine = OpenVinoPaddleOcrEngine.__new__(OpenVinoPaddleOcrEngine)
    engine.minimum_confidence = 0.55
    box = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    other = box.copy()
    other[:, 0] += 10
    engine.detect = lambda _source: [box, other]
    engine.recognize = lambda _crops: [("清楚文字", 0.9), ("模糊文字", 0.3)]
    lines = engine.extract_array(np.zeros((20, 20, 3), dtype=np.uint8))
    assert [line.text for line in lines] == ["清楚文字", "模糊文字"]
    assert [line.searchable for line in lines] == [True, False]


def test_adaptive_tiling_rules_and_edge_alignment():
    wide = np.zeros((1080, 3840, 3), dtype=np.uint8)
    assert should_use_tiles(wide, 0.25, 4, ".png")
    assert not should_use_tiles(wide, 0.70, 100, ".png")
    regions = tile_regions(3840, 1080)
    assert (0, 0, 1280, 960) in regions
    assert any(x + width == 3840 for x, _y, width, _height in regions)
    assert any(y + height == 1080 for _x, y, _width, height in regions)


def test_overlapping_full_and_tile_boxes_are_deduplicated():
    full = np.asarray([[0, 0], [100, 0], [100, 30], [0, 30]], dtype=np.float32)
    tile = np.asarray([[2, 1], [102, 1], [102, 31], [2, 31]], dtype=np.float32)
    results = deduplicate_detections([
        DetectedTextBox(full, 0.9, "full"),
        DetectedTextBox(tile, 0.8, "tile:0,0"),
    ])
    assert len(results) == 1
    assert results[0].source_pass == "tile:0,0"
