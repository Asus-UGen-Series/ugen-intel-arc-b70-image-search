import numpy as np

from visual_search.domain import Asset, ObjectDetection
from visual_search.storage import SQLiteAssetRepository
from visual_search.yolo_openvino import parse_yolo_output, visible_box_mask


def test_object_metadata_is_searchable_in_chinese(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(None, str(tmp_path / "dog.jpg"), 1, 1, "hash", 10, 10, "thumb"))
    repository.replace_objects(asset_id, [ObjectDetection("dog", 0.92, [0, 0, 9, 9], "狗")])
    repository.set_asset_status(asset_id, "ready")
    hit = repository.object_search("找一張有狗的照片")[0]
    assert hit.asset.id == asset_id
    assert hit.object_rank == 1


def test_object_boxes_convert_numpy_scalars_before_json_storage(tmp_path):
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    asset_id = repository.upsert(Asset(None, str(tmp_path / "meter.jpg"), 1, 1, "hash", 10, 10, "thumb"))
    repository.replace_objects(asset_id, [ObjectDetection(
        "multimeter", np.float32(0.91), [np.float32(1), np.float32(2), np.float32(8), np.float32(9)],
    )])
    loaded = repository.object_detections(asset_id)
    assert loaded[0].box == [1.0, 2.0, 8.0, 9.0]


def test_yolo_parser_supports_raw_84_channel_output():
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 100, 80]
    output[0, 4 + 16, 0] = 0.9  # dog
    boxes, scores, classes = parse_yolo_output(output, 0.25)
    assert boxes.shape == (1, 4)
    assert float(scores[0]) > 0.89
    assert classes.tolist() == [16]


def test_yoloe_parser_discards_high_score_zero_area_padding():
    output = np.zeros((1, 300, 38), dtype=np.float32)
    output[0, 0, :6] = [20, 30, 220, 330, 0.91, 1533]
    output[0, 1, :6] = [15, 0, 80, 0, 0.82, 4097]

    boxes, scores, classes = parse_yolo_output(output, 0.20, class_count=4585)

    assert boxes.shape == (1, 4)
    assert float(scores[0]) > 0.90
    assert classes.tolist() == [1533]


def test_yoloe_discards_boxes_that_only_cover_letterbox_padding():
    boxes = np.asarray([
        [20, 40, 120, 100],
        [20, 0, 120, 38],
        [20, 37, 120, 2],
    ], dtype=np.float32)

    visible = visible_box_mask(boxes, pad_x=0, pad_y=40, content_width=640, content_height=360)

    assert visible.tolist() == [True, False, False]
