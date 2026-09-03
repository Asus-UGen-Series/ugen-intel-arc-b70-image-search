from pathlib import Path

import numpy as np
from PIL import Image

from visual_search.domain import OcrLine
from visual_search.gpu_scheduler import GpuRuntimeUnavailable
from visual_search.indexer import IndexController
from visual_search.storage import SQLiteAssetRepository


class FakeOcr:
    model_version = "ocr-v1"

    def __init__(self):
        self.calls = 0

    def extract(self, _path: Path):
        self.calls += 1
        return [OcrLine("紅色筆電", 0.9, [[0, 0], [10, 0], [10, 10], [0, 10]], True)]


class FakeEmbedding:
    model_version = "embedding-v1"
    dimensions = 4

    def __init__(self):
        self.calls = 0

    def image(self, _path: Path):
        self.calls += 1
        return np.asarray([1, 0, 0, 0], dtype=np.float32)

    def text(self, _text: str):
        return np.asarray([1, 0, 0, 0], dtype=np.float32)

    def image_regions(self, _path: Path, boxes):
        return [np.asarray([0, 1, 0, 0], dtype=np.float32) for _box in boxes]


class FakeVectors:
    def __init__(self):
        self.values = {}

    def add(self, key, vector):
        self.values[key] = np.asarray(vector)

    def search(self, _vector, limit):
        return [(key, 1.0) for key in list(self.values)[:limit]]

    def remove(self, keys):
        for key in keys:
            self.values.pop(key, None)

    def save(self):
        pass

    def contains(self, key):
        return key in self.values


def test_incremental_index_skips_unchanged_and_reruns_only_stale_stage(tmp_path):
    root = tmp_path / "中文圖片"
    root.mkdir()
    image = root / "螢幕擷取畫面.png"
    Image.new("RGB", (80, 60), "blue").save(image)
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    vectors = FakeVectors()
    ocr = FakeOcr()
    embedding = FakeEmbedding()
    controller = IndexController(repository, vectors, ocr, embedding, tmp_path / "thumbs")

    controller.scan(root)
    assert repository.count() == 1
    assert ocr.calls == 1
    assert embedding.calls == 1
    controller.scan(root)
    assert ocr.calls == 1
    assert embedding.calls == 1

    upgraded_ocr = FakeOcr()
    upgraded_ocr.model_version = "ocr-v2"
    upgraded = IndexController(repository, vectors, upgraded_ocr, embedding, tmp_path / "thumbs")
    upgraded.scan(root)
    assert upgraded_ocr.calls == 1
    assert embedding.calls == 1


def test_1024_image_builds_four_region_vectors(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (1024, 1024), "blue").save(root / "square.png")
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    controller = IndexController(
        repository, FakeVectors(), FakeOcr(), FakeEmbedding(), tmp_path / "thumbs",
        region_vectors=FakeVectors(),
    )

    controller.scan(root)

    asset = repository.all_assets()[0]
    assert repository.region_count() == 4
    assert repository.stage_states(int(asset.id))["region_embedding"]["status"] == "ready"
    assert asset.status == "ready"


class FatalOcr(FakeOcr):
    def extract(self, _path: Path):
        self.calls += 1
        raise GpuRuntimeUnavailable("Intel GPU runtime fault")


def test_fatal_gpu_error_pauses_scan_without_marking_remaining_images_failed(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for index in range(3):
        Image.new("RGB", (80, 60), "blue").save(root / f"{index}.png")
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    ocr = FatalOcr()
    controller = IndexController(
        repository, FakeVectors(), ocr, FakeEmbedding(), tmp_path / "thumbs",
    )

    controller.scan(root)

    assert ocr.calls == 1
    assert len(repository.all_assets()) == 1
    asset = repository.all_assets()[0]
    assert asset.status == "partial"
    assert repository.stage_states(int(asset.id))["ocr"]["status"] == "pending"
    assert controller.last_stop_reason


def test_remove_root_deletes_only_its_index_and_never_original_images(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_image = first_root / "first.png"
    second_image = second_root / "second.png"
    Image.new("RGB", (80, 60), "blue").save(first_image)
    Image.new("RGB", (80, 60), "green").save(second_image)
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    vectors = FakeVectors()
    controller = IndexController(
        repository, vectors, FakeOcr(), FakeEmbedding(), tmp_path / "thumbs",
    )
    controller.scan(first_root)
    controller.scan(second_root)

    removed = controller.remove_root(first_root, [second_root])

    assert removed == 1
    assert first_image.exists()
    assert second_image.exists()
    assert repository.get_by_path(str(first_image)) is None
    assert repository.get_by_path(str(second_image)) is not None
    assert str(first_root) not in repository.list_roots()
    assert str(second_root) in repository.list_roots()
    assert len(vectors.values) == 1


def test_remove_nested_root_keeps_assets_covered_by_parent_root(tmp_path):
    parent = tmp_path / "library"
    nested = parent / "screenshots"
    nested.mkdir(parents=True)
    image = nested / "screen.png"
    Image.new("RGB", (80, 60), "blue").save(image)
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    controller = IndexController(
        repository, FakeVectors(), FakeOcr(), FakeEmbedding(), tmp_path / "thumbs",
    )
    controller.scan(parent)
    controller.scan(nested)

    removed = controller.remove_root(nested, [parent])

    assert removed == 0
    assert repository.get_by_path(str(image)) is not None
    assert str(parent) in repository.list_roots()
    assert str(nested) not in repository.list_roots()
