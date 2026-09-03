import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QProgressBar

from visual_search.domain import Asset
from visual_search.paths import AppPaths
from visual_search.storage import SQLiteAssetRepository
from visual_search.ui import DeveloperIndexPage, DeveloperSearchPage, MainPage


class VectorPath:
    def __init__(self, path: Path):
        self.path = path


def test_developer_index_tree_and_locations_render(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = AppPaths(
        tmp_path, tmp_path / "metadata.db", tmp_path / "vectors.usearch",
        tmp_path / "thumbnails", tmp_path / "models", tmp_path / "logs",
        tmp_path / "config.json",
    )
    paths.ensure()
    repository = SQLiteAssetRepository(paths.database)
    asset_id = repository.upsert(Asset(
        None, str(tmp_path / "image.png"), 1, 1, "hash", 80, 60,
        str(paths.thumbnails / "thumb.jpg"), status="partial",
    ))
    repository.set_stage(asset_id, "metadata", "ready", "v1", "hash")
    repository.set_stage(asset_id, "embedding", "ready", "v1", "hash")
    page = DeveloperIndexPage(
        repository, paths, VectorPath(tmp_path / "vectors-4.usearch"),
        VectorPath(tmp_path / "regions-4.usearch"),
    )

    assert page.index_tree.columnCount() == 7
    library = page.index_tree.topLevelItem(0)
    assert library.text(0) == "圖片庫（已掃描檔案）"
    assert library.text(1) == "1"
    assert library.childCount() == 5
    assert "metadata.db" in page.index_locations.text()
    assert "vectors-4.usearch" in page.index_locations.text()
    assert "regions-4.usearch" in page.index_locations.text()
    page.close()
    app.processEvents()


def test_per_image_scoring_is_its_own_full_page(tmp_path):
    app = QApplication.instance() or QApplication([])
    repository = SQLiteAssetRepository(tmp_path / "metadata.db")
    page = DeveloperSearchPage(repository)

    assert page.table.columnCount() == 13
    assert page.table.horizontalHeaderItem(0).text() == "Det.#"
    assert page.table.horizontalHeaderItem(5).text() == "VLM"
    assert page.table.horizontalHeaderItem(7).text() == "SigLIP 全圖"
    assert page.table.horizontalHeaderItem(10).text() == "YOLOE"
    assert page.table.isHidden() is False
    page.close()
    app.processEvents()


def test_progress_lifecycle_uses_busy_mode_for_removal_and_hides_afterward():
    app = QApplication.instance() or QApplication([])
    target = SimpleNamespace(
        progress=QProgressBar(), progress_label=QLabel(), _progress_operation=None,
    )

    MainPage._show_busy_progress(target, "正在移除索引", "remove")
    assert target.progress.isVisible()
    assert target.progress.minimum() == 0
    assert target.progress.maximum() == 0
    assert "%" not in target.progress_label.text()

    MainPage._hide_progress(target, "remove")
    assert not target.progress.isVisible()
    assert target.progress.value() == 0
    assert target._progress_operation is None
    app.processEvents()


def test_background_status_waits_for_scan_jobs_and_watchdog_events():
    app = QApplication.instance() or QApplication([])
    target = SimpleNamespace(
        background_index_status=QLabel(), _active_index_jobs=1,
        controller=SimpleNamespace(last_stop_reason=""),
    )
    coverage = {"total_assets": 10, "asset_status": {"ready": 10}}

    MainPage.refresh_background_index_status(target, coverage, {"pending": 0})
    assert "尚未完成" in target.background_index_status.text()
    assert "背景掃描工作 1" in target.background_index_status.text()

    target._active_index_jobs = 0
    MainPage.refresh_background_index_status(target, coverage, {"pending": 2})
    assert "待處理檔案異動 2" in target.background_index_status.text()

    MainPage.refresh_background_index_status(target, coverage, {"pending": 0})
    assert "已完成" in target.background_index_status.text()
    app.processEvents()
