from __future__ import annotations

import logging
import json
import os
import sys
import subprocess
import shutil
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QPointF, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QGridLayout, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget,
    QListWidgetItem,
    QComboBox, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget
)

from .config import AppConfig
from .engines import ColorFallbackEmbedding, NoopOcrEngine, detect_openvino_devices
from .domain import SearchDebugTrace, SearchMode, SearchProgress
from .indexer import IndexController, SearchService
from .library_monitor import LibraryMonitor
from .models import OCR_MODEL_ID, ModelManager, ModelRegistry
from .paths import AppPaths
from .text import metadata_filters

LOGGER = logging.getLogger(__name__)


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal()
    failed = Signal(str)


class IndexWorker(QRunnable):
    def __init__(self, controller: IndexController, root: Path):
        super().__init__()
        self.controller, self.root = controller, root
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.controller.scan(self.root, lambda current, total, name: self.signals.progress.emit(current, total, name))
            self.signals.finished.emit()
        except Exception as exc:
            LOGGER.exception("Indexing failed")
            self.signals.failed.emit(str(exc))


class RootRemovalSignals(QObject):
    finished = Signal(int)
    failed = Signal(str)


class RootRemovalWorker(QRunnable):
    def __init__(self, controller: IndexController, root: Path, keep_roots: list[Path]):
        super().__init__()
        self.controller, self.root, self.keep_roots = controller, root, keep_roots
        self.signals = RootRemovalSignals()

    @Slot()
    def run(self):
        try:
            self.signals.finished.emit(self.controller.remove_root(self.root, self.keep_roots))
        except Exception as exc:
            LOGGER.exception("Unable to remove image root")
            self.signals.failed.emit(str(exc))


class ModelWorkerSignals(QObject):
    progress = Signal(str, object, object)
    finished = Signal()
    failed = Signal(str)


class ModelDownloadWorker(QRunnable):
    def __init__(self, manager: ModelManager):
        super().__init__()
        self.manager = manager
        self.signals = ModelWorkerSignals()

    @Slot()
    def run(self):
        try:
            self.manager.download(lambda name, done, total: self.signals.progress.emit(name, done, total))
            self.signals.progress.emit("本機檔案驗證完成，正在測試模型…", 0, 0)
            if self.manager.model_id == OCR_MODEL_ID:
                from .ocr_openvino import OpenVinoPaddleOcrEngine
                engine = OpenVinoPaddleOcrEngine(self.manager.root)
            elif self.manager.section.startswith("yoloe_"):
                from .yolo_openvino import OpenVinoYoloEngine
                engine = OpenVinoYoloEngine(self.manager.root)
            elif self.manager.section.startswith("vlm_"):
                from PIL import Image
                from .vlm import QwenOpenVinoVlmEngine
                engine = QwenOpenVinoVlmEngine(self.manager.root, "GPU")
                smoke = self.manager.root / ".visual-search-smoke.png"
                Image.new("RGB", (448, 336), (30, 90, 180)).save(smoke)
                try:
                    engine.smoke_test(smoke)
                finally:
                    smoke.unlink(missing_ok=True)
            else:
                from .engines import OpenVinoSiglipEngine
                engine = OpenVinoSiglipEngine(self.manager.root)
            if not self.manager.section.startswith("vlm_"):
                engine.smoke_test()
            self.manager.mark_verified(engine.device)
            self.signals.finished.emit()
        except Exception as exc:
            LOGGER.exception("Model setup failed")
            self.signals.failed.emit(str(exc))


class ModelSetupDialog(QDialog):
    """First-run download; conversion never happens on an end-user PC."""

    def __init__(self, manager: ModelManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.worker = None
        self.setWindowTitle("準備圖片搜尋模型")
        self.setMinimumWidth(620)
        self.setModal(True)
        layout = QVBoxLayout(self)
        if manager.model_id == OCR_MODEL_ID:
            model_title = "正在準備文字辨識模型"
        elif manager.section.startswith("yoloe_"):
            model_title = "正在準備開放物件搜尋模型"
        else:
            model_title = "正在準備智慧圖片搜尋"
        title = QLabel(model_title)
        title.setObjectName("sectionTitle")
        size_gb = manager.total_size / (1024 ** 3)
        detail = QLabel(f"模型總大小約 {size_gb:.1f} GB。會先驗證現有檔案，只下載缺少或損壞的部分。圖片不會上傳。")
        detail.setWordWrap(True)
        self.file_label = QLabel("等待開始…")
        self.file_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.state = QLabel("下載後會自動驗證並測試 Intel GPU／CPU 是否可用。")
        self.state.setWordWrap(True)
        buttons = QHBoxLayout()
        self.retry = QPushButton("開始下載")
        self.retry.setObjectName("primary")
        self.retry.clicked.connect(self.start_download)
        skip = QPushButton("暫時略過")
        self.skip = skip
        skip.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(skip)
        buttons.addWidget(self.retry)
        for widget in (title, detail, self.file_label, self.progress, self.state):
            layout.addWidget(widget)
        layout.addLayout(buttons)
        self.start_download()

    def start_download(self):
        self.retry.setEnabled(False)
        self.skip.setEnabled(False)
        self.state.setText("正在檢查本機模型；驗證通過的檔案會直接重用，接著測試模型是否可用。")
        self.worker = ModelDownloadWorker(self.manager)
        self.worker.signals.progress.connect(self.on_progress)
        self.worker.signals.finished.connect(self.on_finished)
        self.worker.signals.failed.connect(self.on_failed)
        QThreadPool.globalInstance().start(self.worker)

    @Slot(str, object, object)
    def on_progress(self, name: str, done: object, total: object):
        done, total = int(done), int(total)
        self.file_label.setText(f"{name} — {done / 1024**2:.0f} / {total / 1024**2:.0f} MB" if total else name)
        # Per-file bytes are not total setup progress; only the smoke test can
        # declare the package complete.
        self.progress.setRange(0, 0)

    def on_finished(self):
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.state.setText("模型驗證完成，可以開始使用。")
        self.accept()

    def on_failed(self, error: str):
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.state.setText(f"模型尚未就緒：{error}\n可以重試，或暫時使用 OCR 與檔名搜尋。")
        self.retry.setText("重試")
        self.retry.setEnabled(True)
        self.skip.setEnabled(True)

    def reject(self):
        if self.worker is not None and not self.skip.isEnabled():
            return
        super().reject()


class VlmChoiceDialog(QDialog):
    """First-run and settings model selector; never converts models locally."""

    def __init__(self, registry: ModelRegistry, config: AppConfig, paths: AppPaths, parent=None):
        super().__init__(parent)
        self.registry, self.config, self.paths = registry, config, paths
        self.setWindowTitle("選擇最終圖片審核模型")
        self.setMinimumWidth(680)
        layout = QVBoxLayout(self)
        title = QLabel("選擇圖片搜尋的最終審核模式")
        title.setObjectName("sectionTitle")
        description = QLabel(
            "推薦 INT8 會使用約 9.46 GB，適合 Arc Pro B70 32 GB；INT4 約 6.13 GB、速度較快。"
            "模型只在本機執行，圖片不會上傳。"
        )
        description.setWordWrap(True)
        self.choice = QComboBox()
        self.choice.addItem("推薦：Qwen3.5 9B INT8 高品質", "qwen35-9b-int8")
        self.choice.addItem("快速：Qwen3.5 9B INT4", "qwen35-9b-int4")
        self.choice.addItem("從本機 OpenVINO 資料夾／ZIP 匯入", "local")
        self.choice.addItem("稍後設定（先使用混合搜尋）", "off")
        self.state = QLabel("預設使用 INT8；可稍後在「模型與效能」變更。")
        self.state.setWordWrap(True)
        buttons = QHBoxLayout()
        later = QPushButton("稍後設定")
        later.clicked.connect(self.choose_later)
        apply_button = QPushButton("套用並繼續")
        apply_button.setObjectName("primary")
        apply_button.clicked.connect(self.apply_choice)
        buttons.addStretch()
        buttons.addWidget(later)
        buttons.addWidget(apply_button)
        for widget in (title, description, self.choice, self.state):
            layout.addWidget(widget)
        layout.addLayout(buttons)

    def choose_later(self):
        self.config.vlm_policy = "off"
        self.config.vlm_setup_complete = True
        self.config.save(self.paths.config)
        self.accept()

    def apply_choice(self):
        selected = str(self.choice.currentData())
        if selected == "off":
            self.choose_later()
            return
        if selected == "local":
            source = QFileDialog.getExistingDirectory(self, "選擇 OpenVINO VLM 資料夾")
            if not source:
                archive, _ = QFileDialog.getOpenFileName(self, "或選擇 OpenVINO VLM ZIP", "", "ZIP (*.zip)")
                source = archive
            if not source:
                return
            try:
                target = self.registry.import_local_vlm(Path(source))
            except Exception as exc:
                QMessageBox.warning(self, "模型匯入失敗", str(exc))
                return
            self.config.active_vlm_profile = "local"
            self.config.vlm_local_path = str(target)
            self.config.vlm_policy = "auto"
            self.config.vlm_setup_complete = True
            self.config.save(self.paths.config)
            self.accept()
            return
        profile = self.registry.vlm_profile(selected)
        manager = self.registry.vlm_manager(selected)
        free = shutil.disk_usage(self.paths.models).free
        required = self.registry.required_free_space(profile)
        if free < required:
            QMessageBox.warning(
                self, "磁碟空間不足",
                f"{profile.display_name} 需要約 {required / 1024**3:.1f} GB 可用空間，目前只有 {free / 1024**3:.1f} GB。",
            )
            return
        if not manager.ready(verify_hashes=False) and ModelSetupDialog(manager, self).exec() != QDialog.Accepted:
            self.config.active_vlm_profile = selected
            self.choose_later()
            return
        self.config.active_vlm_profile = selected
        self.config.vlm_local_path = ""
        self.config.vlm_policy = "auto"
        self.config.vlm_setup_complete = True
        self.config.save(self.paths.config)
        self.accept()


class SetupPage(QWidget):
    completed = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 70, 80, 70)
        title = QLabel("歡迎使用 Visual Search")
        title.setObjectName("hero")
        intro = QLabel("在電腦上快速找到照片和截圖。所有圖片與搜尋資料都保留在本機。")
        intro.setWordWrap(True)
        self.hardware = QLabel()
        self.hardware.setObjectName("statusCard")
        self.folder = QLineEdit()
        self.folder.setReadOnly(True)
        choose = QPushButton("選擇圖片資料夾")
        choose.clicked.connect(self.choose_folder)
        self.start = QPushButton("開始使用")
        self.start.setObjectName("primary")
        self.start.setEnabled(False)
        self.start.clicked.connect(lambda: self.completed.emit(self.folder.text()))
        for widget in (title, intro, self.hardware, self.folder, choose, self.start):
            layout.addWidget(widget)
        layout.addStretch()
        info = detect_openvino_devices()
        accelerated = str(info.get("selected", "")).startswith("GPU")
        self.hardware.setText("✓ 已啟用硬體加速" if accelerated else "目前使用 CPU；仍可正常使用，更新 Intel 顯示驅動可提升速度。")

    def choose_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "選擇要搜尋的圖片資料夾")
        if selected:
            self.folder.setText(selected)
            self.start.setEnabled(True)


class ResultCard(QFrame):
    def __init__(self, hit):
        super().__init__()
        self.setObjectName("resultCard")
        self.path = hit.asset.path
        layout = QVBoxLayout(self)
        image = QLabel()
        pixmap = QIcon(hit.asset.thumbnail).pixmap(QSize(220, 150))
        image.setPixmap(pixmap)
        image.setAlignment(Qt.AlignCenter)
        name = QLabel(Path(hit.asset.path).name)
        name.setWordWrap(True)
        detail = QLabel(f"符合度 {round(hit.score * 100)}%" + (f" · {hit.asset.ocr_text[:50]}" if hit.asset.ocr_text else ""))
        detail.setObjectName("muted")
        reasons = "\n".join(hit.reasons[:3])
        query_type = {
            "identifier": "程式碼／型號",
            "visual": "畫面描述",
            "mixed": "混合查詢",
            "metadata": "日期／檔案條件",
        }.get(hit.query_type, hit.query_type)
        detail.setText(
            f"排名分數 {round(hit.score * 100)}% · {query_type}"
            + (f" · VLM {round(hit.vlm_score * 100)}" if hit.vlm_score is not None else "")
            + (f"\n{reasons}" if reasons else "")
        )
        detail.setWordWrap(True)
        open_button = QPushButton("開啟原始檔案")
        open_button.clicked.connect(self.open_file)
        for widget in (image, name, detail, open_button):
            layout.addWidget(widget)

    def open_file(self):
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(Path(self.path))])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))


class ResultListModel(QAbstractListModel):
    HitRole = int(Qt.UserRole) + 1

    def __init__(self):
        super().__init__()
        self.hits = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.hits)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.hits):
            return None
        hit = self.hits[index.row()]
        if role == Qt.DecorationRole:
            return QIcon(hit.asset.thumbnail)
        if role == Qt.DisplayRole:
            name = Path(hit.asset.path).name
            score = round(hit.score * 100)
            vlm = f" · VLM {round(hit.vlm_score * 100)}" if hit.vlm_score is not None else ""
            return f"{name}\n排名 {score}%{vlm}"
        if role == Qt.ToolTipRole:
            return "\n".join([hit.asset.path, *hit.reasons[:8]])
        if role == self.HitRole:
            return hit
        return None

    def set_hits(self, hits) -> None:
        self.beginResetModel()
        self.hits = list(hits)
        self.endResetModel()


class StatusCard(QFrame):
    def __init__(self, title: str, state: str, detail: str, tone: str = "ok"):
        super().__init__()
        self.setObjectName(f"statusCard_{tone}")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("statusTitle")
        state_label = QLabel(state)
        state_label.setObjectName(f"statusValue_{tone}")
        state_label.setWordWrap(True)
        detail_label = QLabel(detail)
        detail_label.setObjectName("statusDetail")
        detail_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(state_label)
        layout.addWidget(detail_label)


class SearchSignals(QObject):
    finished = Signal(int, object)
    progress = Signal(int, object)
    failed = Signal(int, str)


class SearchWorker(QRunnable):
    def __init__(self, service: SearchService, query: str, generation: int, mode: SearchMode):
        super().__init__()
        self.service, self.query, self.generation, self.mode = service, query, generation, mode
        self.signals = SearchSignals()

    @Slot()
    def run(self):
        try:
            hits = self.service.search(
                self.query, None,
                lambda update: self.signals.progress.emit(self.generation, update),
                generation=self.generation,
                mode=self.mode,
            )
            self.signals.finished.emit(self.generation, hits)
        except Exception as exc:
            LOGGER.exception("Search failed")
            self.signals.failed.emit(self.generation, str(exc))


class OcrDiagnosticPage(QWidget):
    """Shows exactly what OCR produced and what was admitted to FTS."""

    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        layout = QVBoxLayout(self)
        title_row = QHBoxLayout()
        title = QLabel("OCR 診斷")
        title.setObjectName("sectionTitle")
        refresh = QPushButton("重新整理")
        refresh.clicked.connect(self.refresh)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(refresh)
        explanation = QLabel("綠框代表已進入文字搜尋；橘框代表 OCR 有讀到，但因信心值較低，只保留在診斷資料中。")
        explanation.setObjectName("muted")
        explanation.setWordWrap(True)
        layout.addLayout(title_row)
        layout.addWidget(explanation)

        splitter = QSplitter(Qt.Horizontal)
        self.assets = QListWidget()
        self.assets.setMinimumWidth(280)
        self.assets.currentItemChanged.connect(self.show_asset)
        splitter.addWidget(self.assets)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.summary = QLabel("選擇一張圖片查看 OCR 結果")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.preview = QLabel()
        self.preview.setMinimumSize(640, 360)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #111827; border-radius: 8px;")
        self.table = QTableWidget(0, 7)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setHorizontalHeaderLabels([
            "原始辨識文字", "來源", "偵測信心", "辨識信心", "進入搜尋", "搜尋詞", "座標",
        ])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        right_layout.addWidget(self.summary)
        right_layout.addWidget(self.preview, 2)
        right_layout.addWidget(self.table, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.refresh()

    def refresh(self):
        selected_id = self.assets.currentItem().data(Qt.UserRole) if self.assets.currentItem() else None
        self.assets.clear()
        selected_item = None
        for asset in self.repository.all_assets():
            item = QListWidgetItem(f"[{asset.status}] {Path(asset.path).name}")
            item.setData(Qt.UserRole, asset.id)
            item.setToolTip(asset.path)
            self.assets.addItem(item)
            if asset.id == selected_id:
                selected_item = item
        if selected_item:
            self.assets.setCurrentItem(selected_item)
        elif self.assets.count():
            self.assets.setCurrentRow(0)

    def show_asset(self, current, _previous=None):
        if current is None:
            return
        asset_id = int(current.data(Qt.UserRole))
        asset = self.repository.get(asset_id)
        if asset is None:
            return
        lines = self.repository.ocr_lines(asset_id)
        stages = self.repository.stage_states(asset_id)
        searchable_count = sum(line.searchable for line in lines)
        stage_text = " · ".join(f"{name}:{value.get('status', '?')}" for name, value in sorted(stages.items()))
        self.summary.setText(
            f"{asset.path}\n狀態：{asset.status} · OCR 原始 {len(lines)} 行 · 進入搜尋 {searchable_count} 行"
            + (f"\n階段：{stage_text}" if stage_text else "")
            + (f"\n錯誤：{asset.error}" if asset.error else "")
        )
        self.table.setRowCount(len(lines))
        try:
            diagnostics = json.loads(asset.ocr_diagnostics or "{}")
        except (TypeError, ValueError):
            diagnostics = {}
        if diagnostics:
            self.summary.setText(
                self.summary.text()
                + "\n分塊：{tiling} · full 框：{full} · tile：{tiles} · 去重：{before} → {after} · 全圖縮放：{scale}".format(
                    tiling="啟用" if diagnostics.get("tiling_enabled") else "未啟用",
                    full=diagnostics.get("full_boxes", 0),
                    tiles=diagnostics.get("tile_count", 0),
                    before=diagnostics.get("boxes_before_dedup", 0),
                    after=diagnostics.get("boxes_after_dedup", 0),
                    scale=diagnostics.get("full_scale", "?"),
                )
            )
        for row, line in enumerate(lines):
            values = [
                line.text,
                f"{line.confidence:.3f}",
                "是" if line.searchable else "否（僅保留診斷）",
                str([[round(value, 1) for value in point] for point in line.box]),
            ]
            values = [
                line.text,
                line.source_pass,
                f"{line.detection_confidence:.3f}",
                f"{line.confidence:.3f}",
                "是" if line.searchable else "否（僅保留診斷）",
                " ".join(line.search_terms),
                str([[round(value, 1) for value in point] for point in line.box]),
            ]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if not line.searchable:
                    cell.setForeground(QColor("#b54708"))
                self.table.setItem(row, column, cell)
        self._draw_preview(Path(asset.path), lines)

    def _draw_preview(self, path: Path, lines):
        pixmap = QPixmap()
        try:
            pixmap.loadFromData(path.read_bytes())
        except OSError:
            pixmap = QPixmap()
        if pixmap.isNull():
            self.preview.setText("無法顯示圖片")
            self.preview.setPixmap(QPixmap())
            return
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        for index, line in enumerate(lines, 1):
            if len(line.box) < 4:
                continue
            color = QColor("#12b76a" if line.searchable else "#f79009")
            painter.setPen(QPen(color, max(2, round(pixmap.width() / 700))))
            polygon = QPolygonF([QPointF(float(point[0]), float(point[1])) for point in line.box])
            painter.drawPolygon(polygon)
            painter.drawText(polygon.boundingRect().topLeft() + QPointF(2, -3), str(index))
        painter.end()
        target = self.preview.size() if self.preview.width() > 100 else QSize(900, 520)
        self.preview.setPixmap(pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class DeveloperIndexPage(QWidget):
    """Live index-stage funnel and the exact on-disk index locations."""

    def __init__(self, repository, paths: AppPaths, vectors=None, region_vectors=None):
        super().__init__()
        self.repository = repository
        self.paths = paths
        self.vectors = vectors
        self.region_vectors = region_vectors
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 18)
        layout.setSpacing(10)

        title = QLabel("背景索引漏斗（即時）")
        title.setObjectName("sectionTitle")
        help_text = QLabel(
            "每一關的『完成』代表模型真正跑過；『實際有資料』可區分 OCR／YOLOE 跑完但沒有文字或物件的圖片。"
        )
        help_text.setObjectName("muted")
        help_text.setWordWrap(True)
        self.index_tree = QTreeWidget()
        self.index_tree.setColumnCount(7)
        self.index_tree.setHeaderLabels([
            "處理關卡", "完成", "實際有資料", "待處理", "執行中", "失敗", "補充",
        ])
        self.index_tree.setAlternatingRowColors(True)
        tree_header = self.index_tree.header()
        tree_header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 6):
            tree_header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.index_locations = QLabel()
        self.index_locations.setObjectName("locationBox")
        self.index_locations.setWordWrap(True)
        self.index_locations.setTextInteractionFlags(Qt.TextSelectableByMouse)
        buttons = QHBoxLayout()
        refresh = QPushButton("重新整理索引漏斗")
        refresh.clicked.connect(self.refresh_index_overview)
        open_index = QPushButton("開啟索引資料夾")
        open_index.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.root)))
        )
        buttons.addWidget(refresh)
        buttons.addWidget(open_index)
        buttons.addStretch()
        layout.addWidget(title)
        layout.addWidget(help_text)
        layout.addWidget(self.index_tree, 1)
        layout.addWidget(self.index_locations)
        layout.addLayout(buttons)
        self.refresh_index_overview()

    @staticmethod
    def _stage_values(coverage: dict[str, object], stage: str) -> tuple[int, int, int, int, int]:
        total = int(coverage.get("total_assets", 0))
        states = dict(coverage.get("stages", {}).get(stage, {}))
        ready = int(states.get("ready", 0))
        skipped = int(states.get("skipped", 0))
        running = int(states.get("running", 0))
        error = int(states.get("error", 0))
        explicit_pending = int(states.get("pending", 0))
        known = sum(int(value) for value in states.values())
        pending = explicit_pending + max(0, total - known)
        return ready, skipped, pending, running, error

    @staticmethod
    def _path_status(path: Path) -> str:
        if not path.exists():
            return "尚未建立"
        if path.is_file():
            size = path.stat().st_size
            if size >= 1024 ** 2:
                return f"{size / 1024 ** 2:.1f} MiB"
            return f"{size / 1024:.1f} KiB"
        return "資料夾"

    def refresh_index_overview(self, coverage: dict[str, object] | None = None) -> None:
        coverage = coverage or self.repository.index_coverage()
        total = int(coverage.get("total_assets", 0))
        assets = dict(coverage.get("asset_status", {}))
        self.index_tree.clear()
        library = QTreeWidgetItem([
            "圖片庫（已掃描檔案）", f"{total:,}", f"{total:,}", "—", "—", "—",
            "原始圖片不會複製到索引資料夾",
        ])
        self.index_tree.addTopLevelItem(library)

        def add_stage(parent, label: str, stage: str, actual: int, note: str = ""):
            ready, skipped, pending, running, error = self._stage_values(coverage, stage)
            completed = ready + skipped
            item = QTreeWidgetItem([
                label, f"{completed:,}", f"{actual:,}", f"{pending:,}", f"{running:,}",
                f"{error:,}", note + (f"；略過 {skipped:,}" if skipped else ""),
            ])
            parent.addChild(item)
            return item

        add_stage(library, "1. Metadata／縮圖", "metadata", int(coverage.get("metadata_assets", 0)), "解碼、尺寸、縮圖")
        semantic = QTreeWidgetItem(["2. SigLIP2 語意索引", "", "", "", "", "", ""])
        library.addChild(semantic)
        add_stage(
            semantic, "2a. 全圖 embedding", "embedding", int(coverage.get("full_embeddings", 0)),
            "SQLite 備份＋USearch",
        )
        add_stage(
            semantic, "2b. 局部 embedding", "region_embedding", int(coverage.get("region_assets", 0)),
            f"{int(coverage.get('region_vectors', 0)):,} 個局部向量",
        )
        add_stage(
            library, "3. PP-OCRv6", "ocr", int(coverage.get("ocr_assets", 0)),
            f"{int(coverage.get('ocr_lines', 0)):,} 行文字；0 行也可能是正常完成",
        )
        add_stage(
            library, "4. YOLOE-26", "object", int(coverage.get("object_assets", 0)),
            f"{int(coverage.get('objects', 0)):,} 個框；0 框也可能是正常完成",
        )
        ready_assets = int(assets.get("ready", 0))
        partial_assets = int(assets.get("partial", 0))
        final = QTreeWidgetItem([
            "5. 全部必要關卡完成", f"{ready_assets:,}", f"{ready_assets + partial_assets:,}",
            f"{partial_assets:,}", f"{int(assets.get('indexing', 0)):,}",
            f"{int(assets.get('error', 0)):,}", "partial 仍可搜尋，但背景處理尚未完成",
        ])
        library.addChild(final)
        library.setExpanded(True)
        semantic.setExpanded(True)

        full_vector = Path(getattr(self.vectors, "path", self.paths.vectors))
        region_vector = Path(getattr(
            self.region_vectors, "path", self.paths.vectors.with_name("regions.usearch")
        ))
        wal = Path(str(self.paths.database) + "-wal")
        locations = [
            ("SQLite metadata／OCR／YOLOE／向量備份", self.paths.database),
            ("SQLite WAL", wal),
            ("SigLIP2 全圖 USearch", full_vector),
            ("SigLIP2 局部 USearch", region_vector),
            ("縮圖", self.paths.thumbnails),
            ("模型", self.paths.models),
            ("設定", self.paths.config),
            ("日誌", self.paths.logs),
        ]
        self.index_locations.setText("索引實際位置\n" + "\n".join(
            f"{name}：{path}（{self._path_status(path)}）" for name, path in locations
        ))


class DeveloperSearchPage(QWidget):
    """Search recall/ranking trace. It is never exposed by a frozen production build."""

    CHANNEL_LABELS = {
        "semantic": "SigLIP 全圖",
        "region": "SigLIP 局部",
        "text": "OCR／檔名",
        "object": "YOLOE 物件",
        "metadata": "Metadata",
    }

    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        self.trace: SearchDebugTrace | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 18)
        layout.setSpacing(10)

        title = QLabel("搜尋召回與評分除錯")
        title.setObjectName("sectionTitle")
        warning = QLabel(
            "開發者模式：只會由 quick-test.cmd app 啟用。正式安裝版不顯示此頁，也不保存搜尋 trace。"
        )
        warning.setObjectName("capabilityWarning")
        warning.setWordWrap(True)
        architecture = QLabel(
            "召回層｜SigLIP 全圖找整體畫面；SigLIP 局部找小物件；OCR／檔名找精確文字；"
            "YOLOE 提供固定開放詞彙的軟候選；Metadata 處理日期、格式與資料夾。\n"
            "融合層｜各通道先各取候選，再以加權 RRF 合併；完整檔名、OCR 片語與程式碼識別字另加 bonus。\n"
            "審核層｜visual／mixed 查詢會把動態候選集合全部交給 Qwen3.5；"
            "達到所選門檻的結果全部保留，最後為 70% VLM + 30% deterministic。"
        )
        architecture.setWordWrap(True)
        architecture.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(title)
        layout.addWidget(warning)
        layout.addWidget(architecture)

        self.query_summary = QLabel("完成一次搜尋後，這裡會顯示該次完整 trace。")
        self.query_summary.setWordWrap(True)
        self.query_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.candidate_summary = QLabel()
        self.candidate_summary.setWordWrap(True)
        self.candidate_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.coverage_summary = QLabel()
        self.coverage_summary.setWordWrap(True)
        self.coverage_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.diagnosis = QLabel()
        self.diagnosis.setObjectName("capabilityWarning")
        self.diagnosis.setWordWrap(True)
        for widget in (self.query_summary, self.candidate_summary, self.coverage_summary, self.diagnosis):
            layout.addWidget(widget)

        expected_row = QHBoxLayout()
        self.expected = QLineEdit()
        self.expected.setPlaceholderText("可選：輸入預期應找到的檔名或完整路徑，檢查它在哪一層消失")
        inspect_button = QPushButton("檢查指定圖片")
        inspect_button.clicked.connect(self.inspect_expected)
        expected_row.addWidget(self.expected, 1)
        expected_row.addWidget(inspect_button)
        layout.addLayout(expected_row)
        self.expected_result = QLabel()
        self.expected_result.setWordWrap(True)
        self.expected_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.expected_result)

        self.table = QTableWidget(0, 13)
        self.table.setHorizontalHeaderLabels([
            "Det.#", "檔名", "去向", "最後分數", "Det.分數", "VLM", "Bonus",
            "SigLIP 全圖", "SigLIP 局部", "OCR／檔名", "YOLOE", "Metadata", "命中原因",
        ])
        header = self.table.horizontalHeader()
        for column in (0, 2, 3, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for column in range(7, 12):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(12, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setDefaultSectionSize(64)
        layout.addWidget(self.table, 1)

    @staticmethod
    def _mapping_text(values: dict[str, object]) -> str:
        return "、".join(f"{key}={value}" for key, value in sorted(values.items())) or "無"

    @staticmethod
    def _short_error(value: str) -> str:
        lines = [line.strip() for line in str(value).splitlines() if line.strip()]
        preferred = next(
            (line for line in lines if "CL_" in line or "not JSON serializable" in line or "out of memory" in line.casefold()),
            lines[-1] if lines else "未知錯誤",
        )
        return preferred[:220]

    def _channel_text(self, hit, channel: str) -> str:
        rank = getattr(hit, f"{channel}_rank")
        if rank is None:
            return "—"
        rrf = float(hit.channel_scores.get(channel, 0.0))
        weight = float(self.trace.effective_weights.get(channel, 0.0)) if self.trace else 0.0
        raw = ""
        if channel == "semantic":
            raw = f"sim {hit.semantic_score:.4f}\n"
        elif channel == "region":
            raw = f"sim {hit.region_score:.4f}\n"
        elif channel == "text":
            raw = f"text {hit.text_score:.4f}\n"
        elif channel == "object" and hit.metadata.get("object_confidence") is not None:
            raw = f"conf {float(hit.metadata['object_confidence']):.3f}\n"
        return f"#{rank}\n{raw}RRF {rrf:.4f}\n× {weight:.0%} = {rrf * weight:.4f}"

    def show_trace(self, trace: SearchDebugTrace | None) -> None:
        self.trace = trace
        self.expected_result.clear()
        if trace is None:
            self.query_summary.setText("本次沒有 trace；請確認是從 quick-test.cmd app 啟動。")
            self.table.setRowCount(0)
            return
        type_name = {
            "identifier": "identifier（型號／程式碼）",
            "visual": "visual（畫面描述）",
            "mixed": "mixed（混合）",
            "metadata": "metadata（日期／檔案條件）",
        }.get(trace.query_type, trace.query_type)
        weights = "、".join(
            f"{self.CHANNEL_LABELS.get(name, name)} {weight:.0%}"
            for name, weight in sorted(trace.effective_weights.items())
        ) or "沒有可用通道"
        self.query_summary.setText(
            f"查詢：{trace.query}\n類型：{type_name} · generation {trace.generation} · "
            f"hard filters：{self._mapping_text(trace.filters)}\n"
            f"實際權重：{weights}\n"
            f"公式：{trace.formula}；VLM 啟用時 final = 0.70 × VLM + 0.30 × deterministic"
        )
        counts = trace.candidate_counts
        semantic_dynamic = trace.dynamic_retrieval.get("semantic", {})
        region_dynamic = trace.dynamic_retrieval.get("region", {})
        self.candidate_summary.setText(
            "候選數｜"
            f"全圖 {counts.get('semantic_raw', 0)} → {counts.get('semantic', 0)}；"
            f"局部向量 {counts.get('region_vectors_raw', 0)} → 圖片 {counts.get('region', 0)}；"
            f"文字 {counts.get('text_raw', 0)} → {counts.get('text', 0)}；"
            f"物件 {counts.get('object_raw', 0)} → {counts.get('object', 0)}；"
            f"Metadata {counts.get('metadata', 0)}；完整聯集 {counts.get('union', 0)}；"
            f"VLM {counts.get('vlm_evaluated', 0)}/{counts.get('vlm_input', 0)}；"
            f"最終 {counts.get('final', len(trace.final_results))}。審核來源：{trace.reranker_source}\n"
            f"動態召回｜全圖 cutoff={float(semantic_dynamic.get('cutoff', 0)):.4f}、"
            f"{len(semantic_dynamic.get('rounds', []))} 輪、停止={semantic_dynamic.get('stop_reason', '—')}；"
            f"局部 cutoff={float(region_dynamic.get('cutoff', 0)):.4f}、"
            f"{len(region_dynamic.get('rounds', []))} 輪、停止={region_dynamic.get('stop_reason', '—')}；"
            f"模式={trace.dynamic_retrieval.get('mode', '—')}"
        )
        coverage = trace.index_coverage
        status = coverage.get("asset_status", {})
        stages = coverage.get("stages", {})
        stage_text = "；".join(
            f"{stage}({self._mapping_text(states)})" for stage, states in sorted(stages.items())
        ) or "尚無階段資料"
        self.coverage_summary.setText(
            f"索引覆蓋｜圖片狀態：{self._mapping_text(status)}；全圖向量 {coverage.get('full_embeddings', 0)}；"
            f"局部向量 {coverage.get('region_vectors', 0)}／{coverage.get('region_assets', 0)} 張；"
            f"OCR {coverage.get('ocr_lines', 0)} 行／{coverage.get('ocr_assets', 0)} 張；"
            f"物件 {coverage.get('objects', 0)} 個／{coverage.get('object_assets', 0)} 張。\n階段：{stage_text}"
        )
        stage_errors = coverage.get("stage_errors", [])
        if stage_errors:
            error_text = "；".join(
                f"{item.get('stage', '?')} × {item.get('count', 0)}：{self._short_error(str(item.get('error', '')))}"
                for item in stage_errors[:5]
            )
            self.coverage_summary.setText(self.coverage_summary.text() + "\n主要錯誤：" + error_text)
        notes: list[str] = []
        if int(status.get("partial", 0)) or int(status.get("error", 0)):
            notes.append("索引尚未全數 ready；漏圖可能是對應階段仍 pending／error。")
        if not counts.get("semantic"):
            notes.append("SigLIP 全圖沒有候選：檢查 embedding 階段與全圖 USearch。")
        if not counts.get("region"):
            notes.append("SigLIP 局部沒有候選：小圖可能不建局部向量；大量高解析圖皆為 0 則 region_embedding 尚未完成。")
        if not counts.get("text"):
            notes.append("OCR／檔名沒有候選：精確文字可能未被 OCR 收錄，請搭配 OCR 診斷頁檢查。")
        if trace.filters and any(counts.get(name + "_raw", 0) > counts.get(name, 0) for name in ("semantic", "text", "object")):
            notes.append("hard filter 已移除部分候選；請確認日期、格式或資料夾解析是否符合原意。")
        if trace.reranker_requested and trace.reranker_source == "deterministic-fallback":
            notes.append("VLM 失敗並整次回退，這不是召回漏圖。")
        self.diagnosis.setText("初步判讀：" + (" ".join(notes) if notes else "各通道均有候選；請用下方排名或指定圖片檢查定位。"))

        final_order = {
            hit.asset.id: rank for rank, hit in enumerate(trace.final_results, 1) if hit.asset.id is not None
        }
        self.table.setRowCount(len(trace.deterministic_candidates))
        vlm_input = int(counts.get("vlm_input", 0))
        for row, hit in enumerate(trace.deterministic_candidates):
            asset_id = hit.asset.id
            if asset_id in final_order:
                destination = f"最終 #{final_order[asset_id]}"
            elif hit.vlm_hard_fail:
                destination = "VLM hard_fail 移除"
            elif trace.reranker_requested and row < vlm_input and hit.vlm_score is not None:
                destination = "VLM 分數未達門檻"
            elif trace.reranker_requested and row < vlm_input:
                destination = "VLM 未完成／回退"
            else:
                destination = "Det. 候選，未進最終"
            values = [
                str(row + 1), Path(hit.asset.path).name, destination, f"{hit.score:.4f}",
                f"{hit.deterministic_score:.4f}",
                "—" if hit.vlm_score is None else f"{hit.vlm_score:.2f}", f"{hit.bonus_score:.3f}",
                self._channel_text(hit, "semantic"), self._channel_text(hit, "region"),
                self._channel_text(hit, "text"), self._channel_text(hit, "object"),
                self._channel_text(hit, "metadata"), "\n".join(hit.reasons),
            ]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip(hit.asset.path if column == 1 else value)
                if hit.vlm_hard_fail:
                    cell.setForeground(QColor("#b42318"))
                self.table.setItem(row, column, cell)
        self.table.resizeRowsToContents()

    def inspect_expected(self) -> None:
        needle = self.expected.text().strip().casefold()
        if not needle:
            self.expected_result.setText("請輸入檔名的一部分或完整路徑。")
            return
        matches = [asset for asset in self.repository.all_assets() if needle in asset.path.casefold()]
        if not matches:
            self.expected_result.setText("資料庫中找不到這個檔名／路徑：它尚未被掃描、已被清除，或輸入內容不符。")
            return
        trace = self.trace
        det_order = {
            hit.asset.id: rank for rank, hit in enumerate(trace.deterministic_candidates, 1)
            if hit.asset.id is not None
        } if trace else {}
        final_order = {
            hit.asset.id: rank for rank, hit in enumerate(trace.final_results, 1)
            if hit.asset.id is not None
        } if trace else {}
        details = []
        for asset in matches[:6]:
            asset_id = int(asset.id)
            stages = self.repository.stage_states(asset_id)
            stage_text = "、".join(f"{name}:{state.get('status', '?')}" for name, state in sorted(stages.items())) or "無"
            stage_error_text = "；".join(
                f"{name}：{self._short_error(state.get('error', ''))}"
                for name, state in sorted(stages.items()) if state.get("error")
            )
            ocr_lines = self.repository.ocr_lines(asset_id)
            objects = self.repository.object_detections(asset_id)
            if asset_id in final_order:
                location = f"已進最終 #{final_order[asset_id]}（deterministic #{det_order.get(asset_id, '?')}）"
            elif asset_id in det_order:
                hit = trace.deterministic_candidates[det_order[asset_id] - 1]
                location = (
                    f"有進 deterministic #{det_order[asset_id]}，但未進最終"
                    + ("；VLM hard_fail" if hit.vlm_hard_fail else "；屬於排名／VLM 問題")
                )
            else:
                location = "未進動態 deterministic 候選聯集；屬於召回、索引覆蓋或 hard filter 問題"
            details.append(
                f"{asset.path}\n狀態 {asset.status}；{location}\n"
                f"階段 {stage_text}；OCR {len(ocr_lines)} 行（可搜尋 {sum(line.searchable for line in ocr_lines)}）；"
                f"局部向量 {len(self.repository.region_ids(asset_id))}；物件 {len(objects)}"
                + (f"\n階段錯誤：{stage_error_text}" if stage_error_text else "")
                + (f"；錯誤 {asset.error}" if asset.error else "")
            )
        suffix = f"\n另有 {len(matches) - 6} 張名稱相符，請輸入更完整路徑。" if len(matches) > 6 else ""
        self.expected_result.setText("\n\n".join(details) + suffix)

class MainPage(QWidget):
    def __init__(self, paths: AppPaths, config: AppConfig, controller: IndexController,
                 search: SearchService, model_manager: ModelManager, registry: ModelRegistry):
        super().__init__()
        self.paths, self.config, self.controller, self.search_service = paths, config, controller, search
        self.model_manager = model_manager
        self.registry = registry
        self.developer_mode = (
            os.environ.get("VISUAL_SEARCH_DEVELOPER_MODE", "").strip().casefold() in {"1", "true", "yes", "on"}
            and not getattr(sys, "frozen", False)
        )
        self.developer_diagnostics = None
        self.developer_tab_index = -1
        self.developer_index_page = None
        self.developer_index_tab_index = -1
        if self.developer_mode:
            self.search_service.set_debug_enabled(True)
        self.search_generation = 0
        self._last_result_render = 0.0
        self.pool = QThreadPool.globalInstance()
        self._rescan_queue: list[str] = []
        self._active_index_jobs = 0
        self._progress_operation: str | None = None
        self.library_monitor = LibraryMonitor(controller, config.roots)
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Visual Search")
        title.setObjectName("title")
        self.query = QLineEdit()
        self.query.setPlaceholderText("搜尋：紅色筆電、ASUS UX3405、會議截圖⋯")
        self.query.returnPressed.connect(self.run_search)
        search_button = QPushButton("搜尋")
        search_button.setObjectName("primary")
        search_button.clicked.connect(self.run_search)
        header.addWidget(title)
        header.addWidget(self.query, 1)
        header.addWidget(search_button)
        layout.addLayout(header)
        search_options = QHBoxLayout()
        search_options.addWidget(QLabel("搜尋完整度"))
        self.search_mode_choice = QComboBox()
        self.search_mode_choice.addItem("智慧完整（推薦）", SearchMode.SMART_COMPLETE.value)
        self.search_mode_choice.addItem("徹底檢查全部圖片", SearchMode.EXHAUSTIVE.value)
        mode_index = self.search_mode_choice.findData(config.search_mode)
        self.search_mode_choice.setCurrentIndex(max(0, mode_index))
        search_options.addWidget(self.search_mode_choice)
        search_options.addWidget(QLabel("結果精準度"))
        self.match_threshold_choice = QComboBox()
        self.match_threshold_choice.addItem("較多結果（60分）", 60)
        self.match_threshold_choice.addItem("平衡（70分）", 70)
        self.match_threshold_choice.addItem("較精準（85分）", 85)
        threshold_index = self.match_threshold_choice.findData(config.vlm_match_threshold)
        self.match_threshold_choice.setCurrentIndex(max(0, threshold_index))
        search_options.addWidget(self.match_threshold_choice)
        self.search_mode_choice.currentIndexChanged.connect(self.save_search_preferences)
        self.match_threshold_choice.currentIndexChanged.connect(self.save_search_preferences)
        search_options.addStretch()
        layout.addLayout(search_options)
        self.capability_banner = QLabel()
        self.capability_banner.setWordWrap(True)
        layout.addWidget(self.capability_banner)
        self.search_status = QLabel("輸入圖片內容、畫面中的文字、檔名或物件即可搜尋。")
        self.search_status.setObjectName("muted")
        layout.addWidget(self.search_status)
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_model = ResultListModel()
        self.results_view = QListView()
        self.results_view.setModel(self.results_model)
        self.results_view.setViewMode(QListView.IconMode)
        self.results_view.setResizeMode(QListView.Adjust)
        self.results_view.setMovement(QListView.Static)
        self.results_view.setUniformItemSizes(True)
        self.results_view.setWordWrap(True)
        self.results_view.setIconSize(QSize(220, 150))
        self.results_view.setGridSize(QSize(260, 220))
        self.results_view.setSpacing(8)
        self.results_view.activated.connect(self.open_result)
        self.results_page = QWidget()
        results_layout = QVBoxLayout(self.results_page)
        results_layout.setContentsMargins(0, 8, 0, 0)
        self.background_index_status = QLabel("正在讀取背景圖片分析狀態…")
        self.background_index_status.setWordWrap(True)
        self.background_index_status.setObjectName("capabilityWarning")
        results_layout.addWidget(self.background_index_status)
        results_layout.addWidget(self.results_view, 1)
        self.tabs.addTab(self.results_page, "搜尋結果")
        self.library = QWidget()
        library_layout = QVBoxLayout(self.library)
        self.roots = QListWidget()
        self.roots.addItems(config.roots)
        add = QPushButton("加入圖片資料夾")
        add.clicked.connect(self.add_folder)
        self.remove_folder_button = QPushButton("移除選取資料夾")
        self.remove_folder_button.clicked.connect(self.remove_selected_folder)
        folder_buttons = QHBoxLayout()
        folder_buttons.addWidget(add)
        folder_buttons.addWidget(self.remove_folder_button)
        folder_buttons.addStretch()
        controls = QHBoxLayout()
        self.pause = QPushButton("暫停")
        self.pause.clicked.connect(self.toggle_pause)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.controller.cancel)
        controls.addWidget(self.pause)
        controls.addWidget(cancel)
        rescan = QPushButton("重新掃描所有資料夾")
        rescan.clicked.connect(self.rescan_all)
        controls.addWidget(rescan)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress_label = QLabel("尚未建立索引")
        self.monitor_status = QLabel("異動監看尚未啟動")
        self.monitor_status.setObjectName("muted")
        self.monitor_status.setWordWrap(True)
        for widget in (self.roots, self.progress_label, self.progress, self.monitor_status):
            library_layout.addWidget(widget)
        library_layout.insertLayout(1, folder_buttons)
        library_layout.addLayout(controls)
        self.tabs.addTab(self.library, "圖片資料夾")
        self.ocr_diagnostics = OcrDiagnosticPage(self.controller.repository)
        self.tabs.addTab(self.ocr_diagnostics, "OCR 診斷")
        self.models_page = QWidget()
        self.models_layout = QVBoxLayout(self.models_page)
        self.models_layout.setContentsMargins(14, 14, 14, 18)
        self.models_layout.setSpacing(12)
        self.models_layout.setAlignment(Qt.AlignTop)
        self.models_scroll = QScrollArea()
        self.models_scroll.setObjectName("modelsScroll")
        self.models_scroll.setFrameShape(QFrame.NoFrame)
        self.models_scroll.setWidgetResizable(True)
        self.models_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.models_scroll.setWidget(self.models_page)
        self.tabs.addTab(self.models_scroll, "模型與效能")
        if self.developer_mode:
            self.developer_diagnostics = DeveloperSearchPage(self.controller.repository)
            self.developer_tab_index = self.tabs.addTab(
                self.developer_diagnostics, "每張圖片評分（開發者）"
            )
            self.developer_index_page = DeveloperIndexPage(
                self.controller.repository, self.paths, self.controller.vectors,
                self.controller.region_vectors,
            )
            self.developer_index_tab_index = self.tabs.addTab(
                self.developer_index_page, "索引漏斗（開發者）"
            )
        self.refresh_model_status()
        layout.addWidget(self.tabs)
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.refresh_monitor_status)
        self.monitor_timer.start(3000)
        if config.setup_complete and config.roots:
            QTimer.singleShot(0, self.start_monitor)

    def add_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "加入圖片資料夾")
        selected_key = str(Path(selected).resolve()).casefold() if selected else ""
        existing = {
            str(Path(root).resolve()).casefold() for root in self.config.roots
        }
        if selected and selected_key not in existing:
            self.config.roots.append(selected)
            self.config.save(self.paths.config)
            self.roots.addItem(selected)
            self.library_monitor.add_root(Path(selected))
            self.start_monitor()
            self.start_index(selected)

    def remove_selected_folder(self):
        item = self.roots.currentItem()
        if item is None:
            QMessageBox.information(self, "請選擇資料夾", "請先在清單中選擇要移除的圖片資料夾。")
            return
        selected = item.text()
        answer = QMessageBox.question(
            self, "移除圖片資料夾",
            "這會停止監看此資料夾，並移除它專屬的縮圖、OCR、物件與向量索引。\n\n"
            "原始圖片不會被刪除。確定繼續嗎？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        remaining = [root for root in self.config.roots if root != selected]
        self.stop_monitor()
        self.controller.cancel()
        self.remove_folder_button.setEnabled(False)
        self._show_busy_progress(f"正在移除索引：{selected}", "remove")
        worker = RootRemovalWorker(
            self.controller, Path(selected), [Path(root) for root in remaining],
        )
        self._root_removal_worker = worker
        worker.signals.finished.connect(
            lambda removed, root=selected, roots=remaining: self.folder_removed(root, roots, removed)
        )
        worker.signals.failed.connect(
            lambda error, root=selected: self.folder_remove_failed(root, error)
        )
        self.pool.start(worker)

    def folder_remove_failed(self, selected: str, error: str):
        self.remove_folder_button.setEnabled(True)
        self._hide_progress("remove")
        self.progress_label.setText("資料夾移除失敗")
        self.library_monitor.add_root(Path(selected))
        self.start_monitor()
        QMessageBox.warning(self, "無法移除資料夾", error)

    def folder_removed(self, selected: str, remaining: list[str], removed: int):
        self.config.roots = remaining
        self.config.save(self.paths.config)
        for match in self.roots.findItems(selected, Qt.MatchExactly):
            self.roots.takeItem(self.roots.row(match))
        self._rescan_queue = [root for root in self._rescan_queue if root != selected]
        self.library_monitor.remove_root(Path(selected))
        if remaining:
            self.start_monitor()
        self.remove_folder_button.setEnabled(True)
        self._hide_progress("remove")
        self.progress_label.setText(
            f"已移除資料夾與 {removed:,} 張專屬圖片索引；原始圖片未刪除"
        )
        self.refresh_monitor_status()
        self.ocr_diagnostics.refresh()
        self.refresh_model_status()

    def start_monitor(self):
        for root in self.config.roots:
            self.library_monitor.add_root(Path(root))
        self.library_monitor.start()
        self.refresh_monitor_status()

    def stop_monitor(self):
        self.library_monitor.stop()

    def refresh_monitor_status(self):
        status = self.library_monitor.snapshot()
        state = "正在監看" if status.get("watching") else "未啟用"
        detail = (
            f"{state} {int(status.get('roots', 0))} 個資料夾 · "
            f"待處理異動 {int(status.get('pending', 0))}"
        )
        if status.get("last_scan"):
            detail += f" · 最近校正 {status['last_scan']}"
        if status.get("error"):
            detail += f" · 錯誤：{status['error']}"
        self.monitor_status.setText(detail)
        coverage = self.controller.repository.index_coverage()
        self.refresh_background_index_status(coverage, status)
        if self.developer_index_page is not None:
            self.developer_index_page.refresh_index_overview(coverage)

    def refresh_background_index_status(
        self, coverage: dict[str, object], monitor: dict[str, object] | None = None,
    ):
        assets = dict(coverage.get("asset_status", {}))
        total = int(coverage.get("total_assets", sum(int(value) for value in assets.values())))
        ready = int(assets.get("ready", 0))
        partial = int(assets.get("partial", 0))
        running = int(assets.get("indexing", 0))
        error = int(assets.get("error", 0))
        pending_changes = int((monitor or {}).get("pending", 0))
        active_jobs = int(self._active_index_jobs)
        if total == 0:
            text = "背景圖片分析：尚未建立索引。請先加入圖片資料夾。"
            tone = "capabilityWarning"
        elif ready == total and pending_changes == 0 and active_jobs == 0:
            text = f"✓ 背景圖片分析已完成：{ready:,}/{total:,} 張全部完成。"
            tone = "capabilityOk"
        else:
            text = (
                f"背景圖片分析尚未完成：{ready:,}/{total:,} 張完成 · "
                f"待補 {partial:,} · 處理中 {running:,} · 失敗 {error:,}。"
                f" 待處理檔案異動 {pending_changes:,} · 背景掃描工作 {active_jobs:,}。"
                " 搜尋仍可使用已完成或部分完成的索引。"
            )
            if self.controller.last_stop_reason:
                text += " " + self.controller.last_stop_reason
            tone = "capabilityWarning"
        self.background_index_status.setObjectName(tone)
        self.background_index_status.setText(text)
        self.background_index_status.style().unpolish(self.background_index_status)
        self.background_index_status.style().polish(self.background_index_status)

    def start_index(self, root: str, continue_rescan: bool = False):
        worker = IndexWorker(self.controller, Path(root))
        self._active_index_jobs += 1
        if self._progress_operation != "remove":
            self._show_busy_progress("正在準備圖片分析…", "index")
        worker.signals.progress.connect(self.on_progress)
        worker.signals.failed.connect(self.index_failed)
        worker.signals.finished.connect(self.index_finished)
        if continue_rescan:
            worker.signals.finished.connect(self._start_next_rescan)
        self.pool.start(worker)

    def rescan_all(self):
        self._rescan_queue = [root for root in self.config.roots if Path(root).is_dir()]
        self._start_next_rescan()

    def _start_next_rescan(self):
        if self._rescan_queue:
            self.start_index(self._rescan_queue.pop(0), continue_rescan=True)

    def index_finished(self):
        self._active_index_jobs = max(0, self._active_index_jobs - 1)
        if self._active_index_jobs == 0:
            self._hide_progress("index")
        if self.controller.last_stop_reason:
            self.progress_label.setText(self.controller.last_stop_reason)
            QMessageBox.warning(self, "GPU 索引已安全暫停", self.controller.last_stop_reason)
        else:
            self.progress_label.setText("圖片分析完成")
        self.ocr_diagnostics.refresh()
        self.refresh_model_status()
        self.refresh_monitor_status()

    @Slot(str)
    def index_failed(self, error: str):
        self._active_index_jobs = max(0, self._active_index_jobs - 1)
        if self._active_index_jobs == 0:
            self._hide_progress("index")
        self.progress_label.setText("圖片分析失敗")
        self.refresh_monitor_status()
        QMessageBox.warning(self, "無法分析圖片", error)

    @Slot(int, int, str)
    def on_progress(self, current: int, total: int, name: str):
        if self._progress_operation == "remove":
            return
        self._progress_operation = "index"
        self.progress.setVisible(True)
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        self.progress_label.setText(f"正在分析 {name}（{current}/{total}）")
        self.background_index_status.setObjectName("capabilityWarning")
        self.background_index_status.setText(
            f"背景圖片分析進行中：目前資料夾 {current:,}/{total:,} · {name}"
        )

    def _show_busy_progress(self, text: str, operation: str) -> None:
        """Show activity without inventing a percentage for work with no granular total."""
        self._progress_operation = operation
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.progress_label.setText(text)

    def _hide_progress(self, operation: str) -> None:
        if self._progress_operation != operation:
            return
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._progress_operation = None

    def toggle_pause(self):
        if self.pause.text() == "暫停":
            self.controller.pause()
            self.pause.setText("繼續")
        else:
            self.controller.resume()
            self.pause.setText("暫停")

    def run_search(self):
        query = self.query.text().strip()
        if not query:
            return
        self.results_model.set_hits([])
        mode = SearchMode(str(self.search_mode_choice.currentData()))
        threshold = int(self.match_threshold_choice.currentData())
        self.config.search_mode = mode.value
        self.config.vlm_match_threshold = threshold
        self.config.save(self.paths.config)
        if self.search_service.reranker is not None:
            self.search_service.reranker.match_threshold = threshold
        if mode == SearchMode.EXHAUSTIVE:
            filters = metadata_filters(query)
            total = len(self.controller.repository.eligible_assets(filters))
            answer = QMessageBox.question(
                self, "徹底檢查全部圖片",
                f"這會讓Qwen逐批檢查 {total:,} 張圖片，可能需要很長時間。\n\n確定繼續嗎？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.search_status.setText("正在建立完整候選集合…")
        if self.search_service.scheduler is not None:
            self.search_generation = self.search_service.scheduler.next_generation()
        else:
            self.search_generation += 1
        worker = SearchWorker(self.search_service, query, self.search_generation, mode)
        worker.signals.progress.connect(self.on_search_progress)
        worker.signals.finished.connect(self.on_search_finished)
        worker.signals.failed.connect(self.on_search_failed)
        self.pool.start(worker)

    def save_search_preferences(self, _index=None):
        self.config.search_mode = str(self.search_mode_choice.currentData())
        self.config.vlm_match_threshold = int(self.match_threshold_choice.currentData())
        self.config.save(self.paths.config)
        if self.search_service.reranker is not None:
            self.search_service.reranker.match_threshold = self.config.vlm_match_threshold

    @Slot(int, object)
    def on_search_progress(self, generation: int, update):
        if generation == self.search_generation:
            if isinstance(update, SearchProgress):
                now = time.monotonic()
                if update.complete or now - self._last_result_render >= 0.5:
                    self.results_model.set_hits(update.results)
                    self._last_result_render = now
                suffix = "，排名仍可能變動" if not update.complete else ""
                self.search_status.setText(
                    f"候選 {update.candidate_total:,} · 已審核 {update.reviewed:,} · "
                    f"符合 {update.qualified:,} · 快取 {update.cached:,}{suffix}"
                )

    @Slot(int, object)
    def on_search_finished(self, generation: int, hits):
        if generation != self.search_generation:
            return
        self.results_model.set_hits(hits)
        session = self.search_service.session(generation)
        source = hits[0].rerank_source if hits else (session.reranker_source if session else "deterministic")
        if session is not None and not session.complete:
            self.search_status.setText(
                f"審核未完成：已審核 {session.reviewed:,}/{session.candidate_total:,}，"
                f"保留 {len(hits):,} 張已驗證結果。重新搜尋可從快取續跑。"
            )
        elif source.startswith("qwen3.5"):
            self.search_status.setText(
                f"完整審核完成：{session.candidate_total if session else 0:,} 張候選中有 {len(hits):,} 張符合。"
            )
        else:
            self.search_status.setText(
                f"本次為精確／條件查詢，共找到 {len(hits):,} 張可解釋結果。"
                if hits else "找不到符合的圖片。試試圖片中的文字、檔名或更簡短的描述。"
            )
        if self.developer_diagnostics is not None:
            self.developer_diagnostics.show_trace(self.search_service.debug_trace(generation))
            self.tabs.setTabText(self.developer_tab_index, "每張圖片評分（有新資料）")
        self.refresh_model_status()

    @Slot(int, str)
    def on_search_failed(self, generation: int, error: str):
        if generation != self.search_generation:
            return
        self.results_model.set_hits([])
        self.search_status.setText("搜尋發生錯誤，請到模型與效能頁查看狀態。")
        if self.developer_diagnostics is not None:
            self.developer_diagnostics.query_summary.setText(f"搜尋在建立完整 trace 前失敗：{error}")
        self.refresh_model_status()

    @Slot(QModelIndex)
    def open_result(self, index: QModelIndex):
        hit = self.results_model.data(index, ResultListModel.HitRole)
        if hit is None:
            return
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(Path(hit.asset.path))])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(hit.asset.path))

    @staticmethod
    def _package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "未安裝"

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                MainPage._clear_layout(item.layout())

    def refresh_model_status(self):
        self._clear_layout(self.models_layout)

        title = QLabel("模型與效能狀態")
        title.setObjectName("sectionTitle")
        explanation = QLabel("這裡會直接告訴你哪些模型已可使用，以及目前能使用哪些搜尋功能。")
        explanation.setObjectName("muted")
        explanation.setWordWrap(True)
        self.models_layout.addWidget(title)
        self.models_layout.addWidget(explanation)

        top_buttons = QHBoxLayout()
        refresh = QPushButton("重新檢查狀態")
        refresh.clicked.connect(self.refresh_model_status)
        open_folder = QPushButton("開啟模型資料夾")
        open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.models)))
        )
        top_buttons.addWidget(refresh)
        top_buttons.addWidget(open_folder)
        top_buttons.addStretch()
        self.models_layout.addLayout(top_buttons)

        object_settings = QHBoxLayout()
        object_label = QLabel("物件模型")
        self.object_profile_choice = QComboBox()
        self.object_profile_choice.addItem("YOLOE-26S Prompt-Free（推薦）", "yoloe26s-pf")
        self.object_profile_choice.addItem("YOLOE-26M Prompt-Free（高準確）", "yoloe26m-pf")
        object_index = self.object_profile_choice.findData(self.config.active_object_profile)
        self.object_profile_choice.setCurrentIndex(max(0, object_index))
        apply_object = QPushButton("套用物件模型")
        apply_object.clicked.connect(self.apply_object_settings)
        import_object = QPushButton("匯入本機模型…")
        import_object.clicked.connect(self.import_object_model)
        object_settings.addWidget(object_label)
        object_settings.addWidget(self.object_profile_choice, 1)
        object_settings.addWidget(import_object)
        object_settings.addWidget(apply_object)
        self.models_layout.addLayout(object_settings)
        self.object_model_path = QLabel()
        self.object_model_path.setObjectName("modelPath")
        self.object_model_path.setWordWrap(True)
        self.object_model_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.object_profile_choice.currentIndexChanged.connect(self.update_object_model_path)
        self.models_layout.addWidget(self.object_model_path)
        self.update_object_model_path()

        settings_row = QHBoxLayout()
        self.vlm_profile_choice = QComboBox()
        self.vlm_profile_choice.addItem("Qwen3.5 9B INT8（推薦）", "qwen35-9b-int8")
        self.vlm_profile_choice.addItem("Qwen3.5 9B INT4（快速）", "qwen35-9b-int4")
        self.vlm_profile_choice.addItem("本機 OpenVINO 模型", "local")
        profile_index = self.vlm_profile_choice.findData(self.config.active_vlm_profile)
        self.vlm_profile_choice.setCurrentIndex(max(0, profile_index))
        self.vlm_policy_choice = QComboBox()
        self.vlm_policy_choice.addItem("自動：只審核畫面描述／混合查詢", "auto")
        self.vlm_policy_choice.addItem("永遠啟用", "always")
        self.vlm_policy_choice.addItem("停用 VLM", "off")
        policy_index = self.vlm_policy_choice.findData(self.config.vlm_policy)
        self.vlm_policy_choice.setCurrentIndex(max(0, policy_index))
        apply_vlm = QPushButton("套用 VLM 設定")
        apply_vlm.clicked.connect(self.apply_vlm_settings)
        settings_row.addWidget(self.vlm_profile_choice, 2)
        settings_row.addWidget(self.vlm_policy_choice, 2)
        settings_row.addWidget(apply_vlm)
        self.models_layout.addLayout(settings_row)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        info = detect_openvino_devices()
        accelerated = str(info.get("selected", "")).startswith("GPU")
        cards.addWidget(StatusCard(
            "硬體加速",
            "已使用 Intel GPU" if accelerated else "目前使用 CPU",
            f"可用裝置：{', '.join(info.get('available', [])) or 'CPU'}",
            "ok" if accelerated else "warning",
        ), 0, 0)

        ocr = self.controller.ocr
        ocr_runtime = getattr(ocr, "runtime_details", lambda: [])()
        execution = sorted({device for item in ocr_runtime for device in item.get("execution_devices", [])})
        openvino_ocr = bool(ocr_runtime) and all(item.get("backend") == "OpenVINO FP16" for item in ocr_runtime)
        if isinstance(ocr, NoopOcrEngine):
            ocr_state, ocr_detail, ocr_tone = "不可用", "OCR runtime 沒有載入，圖片文字不會進入搜尋索引。", "error"
        elif openvino_ocr and any(str(device).startswith("GPU") for device in execution):
            ocr_state, ocr_detail, ocr_tone = "OpenVINO 已使用 Intel GPU", "偵測與辨識實際裝置：" + ", ".join(execution), "ok"
        else:
            ocr_state, ocr_detail, ocr_tone = "CPU 相容模式", "目前 OCR 不是 B70 加速；安裝 OpenVINO PP-OCRv6 模型後會切換 GPU。", "warning"
        cards.addWidget(StatusCard("圖片文字辨識", ocr_state, ocr_detail, ocr_tone), 0, 1)

        semantic_ready = not isinstance(self.search_service.embeddings, ColorFallbackEmbedding)
        if semantic_ready:
            semantic_state = "SigLIP 2 NaFlex 已載入"
            semantic_detail = f"SO400M、256 patches、FP16 OpenVINO；執行裝置：{getattr(self.search_service.embeddings, 'device', 'AUTO')}"
        elif self.model_manager.configured():
            semantic_state = "尚未下載或載入失敗"
            semantic_detail = "重新啟動程式會自動續傳；完成後會先做實際推論測試。"
        else:
            semantic_state = "發布來源尚未設定"
            semantic_detail = "開發者需先轉換模型並執行 publish-model.py，將固定版本與校驗值寫入安裝包。"
        cards.addWidget(StatusCard(
            "圖片語意搜尋",
            semantic_state,
            semantic_detail,
            "ok" if semantic_ready else "warning",
        ), 1, 0)

        object_runtime = getattr(self.controller.objects, "runtime_details", lambda: [])()
        object_devices = sorted({device for item in object_runtime for device in item.get("execution_devices", [])})
        object_ready = self.controller.objects is not None
        object_details = object_runtime[0] if object_runtime else {}
        coverage = self.controller.repository.index_coverage()
        object_stats = coverage.get("object_stats", {})
        vocabulary_size = int(object_details.get("vocabulary_size", 0) or 0)
        object_name = str(object_details.get("scale", self.config.active_object_profile)).upper()
        active_object_manager = self.registry.object_manager(self.config.active_object_profile)
        try:
            self.registry.validate_object_directory(active_object_manager.root, self.config.active_object_profile)
            object_package_ready = True
        except (OSError, ValueError):
            object_package_ready = active_object_manager.ready(verify_hashes=False)
        if object_ready:
            object_state = f"YOLOE-26 {object_name} Prompt-Free 已載入"
            object_tone = "ok" if any(str(item).startswith("GPU") for item in object_devices) else "warning"
        elif object_package_ready:
            object_state = "模型已安裝，重新啟動後載入"
            object_tone = "neutral"
        else:
            object_state = "尚未安裝"
            object_tone = "warning"
        object_detail = (
            f"固定開放詞彙：{vocabulary_size:,} · 實際裝置：{', '.join(object_devices) or '—'}\n"
            f"已推論 {int(object_stats.get('ready', 0)):,}/{int(object_stats.get('total', 0)):,} · "
            f"待處理 {int(object_stats.get('pending', 0)):,} · 執行中 {int(object_stats.get('running', 0)):,} · "
            f"失敗 {int(object_stats.get('error', 0)):,}\n"
            f"有偵測 {int(object_stats.get('with_detections', 0)):,} 張 · "
            f"零偵測 {int(object_stats.get('zero_detections', 0)):,} 張 · "
            f"偵測框 {int(object_stats.get('detections', 0)):,} 個"
            if object_ready else f"預期位置：{active_object_manager.root}"
        )
        cards.addWidget(StatusCard(
            "物件辨識",
            object_state,
            object_detail,
            object_tone,
        ), 2, 0)

        vlm_runtime = getattr(self.search_service.reranker.engine, "runtime_details", lambda: [])() if self.search_service.reranker else []
        vlm_installed = bool(vlm_runtime and (vlm_runtime[0].get("installed") or vlm_runtime[0].get("loaded")))
        vlm_loaded = any(bool(item.get("loaded")) for item in vlm_runtime)
        vlm_devices = sorted({device for item in vlm_runtime for device in item.get("execution_devices", [])})
        vlm_activity = (
            self.search_service.reranker.execution_status()
            if self.search_service.reranker and callable(getattr(self.search_service.reranker, "execution_status", None))
            else {"state": "尚未執行", "detail": "沒有 VLM 執行紀錄"}
        )
        if self.config.vlm_policy == "off":
            vlm_state, vlm_tone = "已停用", "warning"
        elif vlm_loaded:
            vlm_state, vlm_tone = "Qwen3.5 已載入", "ok"
        elif vlm_installed:
            vlm_state, vlm_tone = "Qwen3.5 已安裝，尚未載入", "neutral"
        else:
            vlm_state, vlm_tone = "尚未安裝", "warning"
        activity_detail = (
            f"最近一次：{vlm_activity.get('state', '尚未執行')} · "
            f"候選 {int(vlm_activity.get('candidate_count', 0)):,} · "
            f"模型實際推論 {int(vlm_activity.get('model_reviewed', 0)):,} · "
            f"快取 {int(vlm_activity.get('cache_hits', 0)):,} · "
            f"批次 {int(vlm_activity.get('batch_count', 0)):,}"
        )
        if vlm_activity.get("updated_at"):
            activity_detail += f" · {vlm_activity['updated_at']}"
        activity_detail += "\n" + str(vlm_activity.get("detail", ""))
        if vlm_activity.get("error"):
            activity_detail += "\n錯誤：" + str(vlm_activity["error"])
        cards.addWidget(StatusCard(
            "最終圖片審核",
            vlm_state,
            f"模式：{self.config.vlm_policy} · 實際裝置：{', '.join(vlm_devices) or '尚未建立推論工作'}\n"
            + activity_detail,
            vlm_tone,
        ), 2, 1)

        indexed = self.controller.repository.count()
        cards.addWidget(StatusCard(
            "圖片索引",
            f"已完成 {indexed:,} 張",
            f"局部向量 {self.controller.repository.region_count():,} 個 · OpenVINO {self._package_version('openvino')}",
            "ok" if indexed else "neutral",
        ), 1, 1)
        self.models_layout.addLayout(cards)

        runtime_title = QLabel("各模型實際執行裝置")
        runtime_title.setObjectName("sectionTitle")
        self.models_layout.addWidget(runtime_title)
        runtime_grid = QGridLayout()
        runtime_grid.setHorizontalSpacing(12)
        runtime_grid.setVerticalSpacing(12)
        runtime_grid.setColumnStretch(0, 1)
        runtime_grid.setColumnStretch(1, 1)
        semantic_runtime = getattr(self.search_service.embeddings, "runtime_details", lambda: [])()
        for index, item in enumerate(vlm_runtime + object_runtime + semantic_runtime + ocr_runtime):
            devices = [str(device) for device in item.get("execution_devices", [])]
            on_gpu = any(device.startswith("GPU") for device in devices)
            loaded = bool(item.get("loaded"))
            state = "Intel GPU" if on_gpu else ("CPU／未加速" if loaded else "未載入")
            detail = (
                f"Runtime：{item.get('backend', '未知')} · "
                f"指定：{item.get('requested_device', '—')} · "
                f"實際：{', '.join(devices) or '—'}"
            )
            runtime_grid.addWidget(
                StatusCard(str(item.get("name", "模型")), state, detail, "ok" if on_gpu else "warning"),
                index // 2,
                index % 2,
            )
        self.models_layout.addLayout(runtime_grid)

        locations = QLabel(
            "模型根目錄：" + str(self.paths.models) + "\n"
            "SigLIP2：" + str(self.model_manager.root) + "\n"
            "PP-OCRv6：" + str(self.registry.manager("ocr_model").root) + "\n"
            "YOLOE-26S：" + str(self.registry.object_manager("yoloe26s-pf").root) + "\n"
            "YOLOE-26M：" + str(self.registry.object_manager("yoloe26m-pf").root) + "\n"
            "Qwen3.5 INT8：" + str(self.registry.vlm_manager("qwen35-9b-int8").root) + "\n"
            "索引與設定：" + str(self.paths.root)
        )
        locations.setObjectName("locationBox")
        locations.setWordWrap(True)
        locations.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.models_layout.addWidget(locations)
        legacy_yolo = self.paths.models / "yolo11m-openvino-fp16"
        if legacy_yolo.exists():
            legacy_notice = QLabel(
                "偵測到舊版 YOLO11m 模型。Visual Search 不會自動刪除；確認 YOLOE 可用後可手動清理此資料夾。"
            )
            legacy_notice.setObjectName("muted")
            legacy_notice.setWordWrap(True)
            self.models_layout.addWidget(legacy_notice)

        if semantic_ready:
            self.capability_banner.setObjectName("capabilityOk")
            self.capability_banner.setText("✓ 圖片語意搜尋與文字搜尋都可以使用")
        else:
            self.capability_banner.setObjectName("capabilityWarning")
            self.capability_banner.setText("目前模式：可搜尋圖片中的文字與檔名。圖片語意模型尚未安裝，自然語言內容搜尋暫不可用。")
        self.capability_banner.style().unpolish(self.capability_banner)
        self.capability_banner.style().polish(self.capability_banner)

    def update_object_model_path(self):
        if not hasattr(self, "object_profile_choice") or not hasattr(self, "object_model_path"):
            return
        profile_id = str(self.object_profile_choice.currentData())
        manager = self.registry.object_manager(profile_id)
        try:
            self.registry.validate_object_directory(manager.root, profile_id)
            state = "已安裝"
        except (OSError, ValueError):
            state = "尚未安裝"
        self.object_model_path.setText(f"{state} · 模型資料夾：{manager.root}")

    def import_object_model(self, _checked: bool = False) -> bool:
        profile_id = str(self.object_profile_choice.currentData())
        source = QFileDialog.getExistingDirectory(self, "選擇 YOLOE-26 OpenVINO 模型資料夾")
        if not source:
            archive, _ = QFileDialog.getOpenFileName(
                self, "或選擇 YOLOE-26 OpenVINO ZIP", "", "ZIP (*.zip)"
            )
            source = archive
        if not source:
            return False
        try:
            target = self.registry.import_local_object(Path(source), profile_id)
        except Exception as exc:
            QMessageBox.warning(self, "模型匯入失敗", str(exc))
            return False
        self.config.active_object_profile = profile_id
        self.config.save(self.paths.config)
        QMessageBox.information(
            self,
            "模型匯入完成",
            f"模型已安全複製到：\n{target}\n\n重新啟動 Visual Search 後會以 OpenVINO 載入。",
        )
        self.refresh_model_status()
        return True

    def apply_object_settings(self):
        profile_id = str(self.object_profile_choice.currentData())
        manager = self.registry.object_manager(profile_id)
        try:
            self.registry.validate_object_directory(manager.root, profile_id)
            local_ready = True
        except (OSError, ValueError):
            local_ready = False
        if not local_ready and not manager.ready(verify_hashes=False):
            if manager.configured():
                if ModelSetupDialog(manager, self).exec() != QDialog.Accepted:
                    return
            else:
                answer = QMessageBox.question(
                    self,
                    "尚未安裝物件模型",
                    f"找不到所選 YOLOE-26 OpenVINO 模型。\n\n預期位置：\n{manager.root}\n\n"
                    "若你已完成轉換或下載模型，可以現在匯入資料夾或 ZIP。要開始匯入嗎？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer != QMessageBox.Yes or not self.import_object_model():
                    return
                return
        self.config.active_object_profile = profile_id
        self.config.save(self.paths.config)
        QMessageBox.information(
            self,
            "設定已儲存",
            "重新啟動後會使用所選 YOLOE 模型；既有圖片只會補跑物件階段。",
        )

    def apply_vlm_settings(self):
        profile_id = str(self.vlm_profile_choice.currentData())
        policy = str(self.vlm_policy_choice.currentData())
        if profile_id == "local":
            source = QFileDialog.getExistingDirectory(self, "選擇 OpenVINO VLM 資料夾")
            if not source:
                archive, _ = QFileDialog.getOpenFileName(self, "或選擇 OpenVINO VLM ZIP", "", "ZIP (*.zip)")
                source = archive
            if not source:
                return
            try:
                target = self.registry.import_local_vlm(Path(source))
            except Exception as exc:
                QMessageBox.warning(self, "模型匯入失敗", str(exc))
                return
            self.config.vlm_local_path = str(target)
        elif policy != "off":
            manager = self.registry.vlm_manager(profile_id)
            if not manager.ready(verify_hashes=False) and ModelSetupDialog(manager, self).exec() != QDialog.Accepted:
                return
            self.config.vlm_local_path = ""
        self.config.active_vlm_profile = profile_id
        self.config.vlm_policy = policy
        self.config.vlm_setup_complete = True
        self.config.save(self.paths.config)
        QMessageBox.information(self, "設定已儲存", "模型設定已儲存，重新啟動 Visual Search 後生效。")


class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths, config: AppConfig, controller: IndexController,
                 search: SearchService, model_manager: ModelManager, registry: ModelRegistry):
        super().__init__()
        self.paths, self.config, self.controller, self.search = paths, config, controller, search
        self.setWindowTitle("Visual Search")
        self.resize(1120, 760)
        self.pages = QStackedWidget()
        self.setup = SetupPage()
        self.setup.completed.connect(self.finish_setup)
        self.main = MainPage(paths, config, controller, search, model_manager, registry)
        self.pages.addWidget(self.setup)
        self.pages.addWidget(self.main)
        self.setCentralWidget(self.pages)
        self.pages.setCurrentWidget(self.main if config.setup_complete else self.setup)
        self.setStyleSheet(STYLES)
        if config.setup_complete and any(Path(root).is_dir() for root in config.roots):
            QTimer.singleShot(0, self.main.rescan_all)

    def finish_setup(self, folder: str):
        self.config.setup_complete = True
        self.config.roots = [folder]
        self.config.save(self.paths.config)
        self.main.roots.clear()
        self.main.roots.addItems(self.config.roots)
        self.pages.setCurrentWidget(self.main)
        self.main.library_monitor.add_root(Path(folder))
        self.main.start_monitor()
        self.main.start_index(folder)

    def closeEvent(self, event):
        self.controller.cancel()
        if self.controller.scheduler is not None:
            self.controller.scheduler.next_generation()
        self.main.stop_monitor()
        super().closeEvent(event)


STYLES = """
QWidget { font-family: 'Segoe UI'; font-size: 14px; color: #172033; background: #f7f8fb; }
QLabel { background: transparent; }
QLineEdit { background: white; border: 1px solid #d7dce5; border-radius: 9px; padding: 11px; }
QComboBox { background: white; border: 1px solid #d7dce5; border-radius: 8px; padding: 9px 12px; min-height: 20px; }
QPushButton { background: white; border: 1px solid #ccd3df; border-radius: 9px; padding: 10px 16px; }
QPushButton:hover { border-color: #315efb; }
QPushButton#primary { background: #315efb; color: white; border: none; font-weight: 600; }
QLabel#hero { font-size: 32px; font-weight: 700; color: #101828; }
QLabel#title { font-size: 22px; font-weight: 700; padding-right: 18px; }
QLabel#muted { color: #667085; font-size: 12px; }
QLabel#statusCard { background: #ecfdf3; border-radius: 10px; padding: 16px; color: #067647; }
QLabel#sectionTitle { font-size: 22px; font-weight: 700; margin-top: 8px; }
QLabel#capabilityOk { background: #ecfdf3; color: #067647; border-radius: 8px; padding: 10px 14px; }
QLabel#capabilityWarning { background: #fffaeb; color: #93370d; border-radius: 8px; padding: 10px 14px; }
QLabel#statusTitle { color: #667085; font-size: 12px; font-weight: 600; }
QLabel#statusValue_ok { color: #067647; font-size: 19px; font-weight: 700; }
QLabel#statusValue_warning { color: #b54708; font-size: 19px; font-weight: 700; }
QLabel#statusValue_error { color: #b42318; font-size: 19px; font-weight: 700; }
QLabel#statusValue_neutral { color: #344054; font-size: 19px; font-weight: 700; }
QLabel#statusDetail { color: #667085; font-size: 12px; }
QLabel#modelPath { color: #667085; font-size: 12px; padding: 0 2px 6px 2px; }
QLabel#locationBox { background: white; border: 1px solid #e4e7ec; border-radius: 10px; padding: 14px; color: #475467; }
QFrame#resultCard { background: white; border: 1px solid #e4e7ec; border-radius: 12px; min-width: 220px; max-width: 260px; }
QFrame#statusCard_ok, QFrame#statusCard_warning, QFrame#statusCard_error, QFrame#statusCard_neutral { background: white; border: 1px solid #e4e7ec; border-radius: 12px; min-height: 118px; }
QScrollArea#modelsScroll { border: 0; background: #f7f8fb; }
QTabWidget::pane { border: 0; }
"""
