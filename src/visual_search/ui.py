from __future__ import annotations

import logging
import json
import os
import subprocess
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRunnable, QSize, Qt, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QGridLayout, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QComboBox, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget
)

from .config import AppConfig
from .engines import ColorFallbackEmbedding, NoopOcrEngine, detect_openvino_devices
from .indexer import IndexController, SearchService
from .models import OCR_MODEL_ID, ModelManager, ModelRegistry
from .paths import AppPaths

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
        detail = QLabel(f"第一次使用需要下載約 {size_gb:.1f} GB。模型只下載一次，圖片不會上傳。")
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
        required = int(self.manager.total_size * 1.2) + 2 * 1024 ** 3
        free = shutil.disk_usage(self.manager.root.parent).free
        if free < required:
            self.on_failed(f"磁碟空間不足：需要約 {required / 1024**3:.1f} GB，目前可用 {free / 1024**3:.1f} GB。")
            return
        self.retry.setEnabled(False)
        self.skip.setEnabled(False)
        self.state.setText("正在下載並校驗模型，請保持網路連線…")
        self.worker = ModelDownloadWorker(self.manager)
        self.worker.signals.progress.connect(self.on_progress)
        self.worker.signals.finished.connect(self.on_finished)
        self.worker.signals.failed.connect(self.on_failed)
        QThreadPool.globalInstance().start(self.worker)

    @Slot(str, object, object)
    def on_progress(self, name: str, done: object, total: object):
        done, total = int(done), int(total)
        self.file_label.setText(f"{name} — {done / 1024**2:.0f} / {total / 1024**2:.0f} MB" if total else name)
        self.progress.setValue(round(done / total * 1000) if total else 0)

    def on_finished(self):
        self.progress.setValue(1000)
        self.state.setText("模型驗證完成，可以開始使用。")
        self.accept()

    def on_failed(self, error: str):
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


class StatusCard(QFrame):
    def __init__(self, title: str, state: str, detail: str, tone: str = "ok"):
        super().__init__()
        self.setObjectName(f"statusCard_{tone}")
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setObjectName("statusTitle")
        state_label = QLabel(state)
        state_label.setObjectName(f"statusValue_{tone}")
        detail_label = QLabel(detail)
        detail_label.setObjectName("statusDetail")
        detail_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(state_label)
        layout.addWidget(detail_label)


class SearchSignals(QObject):
    finished = Signal(int, object)
    progress = Signal(int, int, int)
    failed = Signal(int, str)


class SearchWorker(QRunnable):
    def __init__(self, service: SearchService, query: str, generation: int):
        super().__init__()
        self.service, self.query, self.generation = service, query, generation
        self.signals = SearchSignals()

    @Slot()
    def run(self):
        try:
            hits = self.service.search(
                self.query, 10,
                lambda current, total: self.signals.progress.emit(self.generation, current, total),
                generation=self.generation,
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

class MainPage(QWidget):
    def __init__(self, paths: AppPaths, config: AppConfig, controller: IndexController,
                 search: SearchService, model_manager: ModelManager, registry: ModelRegistry):
        super().__init__()
        self.paths, self.config, self.controller, self.search_service = paths, config, controller, search
        self.model_manager = model_manager
        self.registry = registry
        self.search_generation = 0
        self.pool = QThreadPool.globalInstance()
        self._rescan_queue: list[str] = []
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
        self.capability_banner = QLabel()
        self.capability_banner.setWordWrap(True)
        layout.addWidget(self.capability_banner)
        self.search_status = QLabel("輸入圖片內容、畫面中的文字、檔名或物件即可搜尋。")
        self.search_status.setObjectName("muted")
        layout.addWidget(self.search_status)
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.results_widget)
        self.tabs.addTab(scroll, "搜尋結果")
        self.library = QWidget()
        library_layout = QVBoxLayout(self.library)
        self.roots = QListWidget()
        self.roots.addItems(config.roots)
        add = QPushButton("加入圖片資料夾")
        add.clicked.connect(self.add_folder)
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
        self.progress_label = QLabel("尚未建立索引")
        for widget in (self.roots, add, self.progress_label, self.progress):
            library_layout.addWidget(widget)
        library_layout.addLayout(controls)
        self.tabs.addTab(self.library, "圖片資料夾")
        self.ocr_diagnostics = OcrDiagnosticPage(self.controller.repository)
        self.tabs.addTab(self.ocr_diagnostics, "OCR 診斷")
        self.models_page = QWidget()
        self.models_layout = QVBoxLayout(self.models_page)
        self.models_layout.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self.models_page, "模型與效能")
        self.refresh_model_status()
        layout.addWidget(self.tabs)

    def add_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "加入圖片資料夾")
        if selected and selected not in self.config.roots:
            self.config.roots.append(selected)
            self.config.save(self.paths.config)
            self.roots.addItem(selected)
            self.start_index(selected)

    def start_index(self, root: str, continue_rescan: bool = False):
        worker = IndexWorker(self.controller, Path(root))
        worker.signals.progress.connect(self.on_progress)
        worker.signals.failed.connect(lambda error: QMessageBox.warning(self, "無法分析圖片", error))
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
        self.progress_label.setText("圖片分析完成")
        self.ocr_diagnostics.refresh()
        self.refresh_model_status()

    @Slot(int, int, str)
    def on_progress(self, current: int, total: int, name: str):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(current)
        self.progress_label.setText(f"正在分析 {name}（{current}/{total}）")

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
        self._clear_layout(self.results_layout)
        self.results_layout.addWidget(QLabel("正在建立候選結果並進行最終審核…"), 0, 0, 1, 4)
        if self.search_service.scheduler is not None:
            self.search_generation = self.search_service.scheduler.next_generation()
        else:
            self.search_generation += 1
        worker = SearchWorker(self.search_service, query, self.search_generation)
        worker.signals.progress.connect(self.on_search_progress)
        worker.signals.finished.connect(self.on_search_finished)
        worker.signals.failed.connect(self.on_search_failed)
        self.pool.start(worker)

    @Slot(int, int, int)
    def on_search_progress(self, generation: int, current: int, total: int):
        if generation == self.search_generation:
            self.search_status.setText(f"初步找到 {total} 張，Qwen3.5 正在審核 {current}/{total}…")

    @Slot(int, object)
    def on_search_finished(self, generation: int, hits):
        if generation != self.search_generation:
            return
        self._clear_layout(self.results_layout)
        if not hits:
            self.results_layout.addWidget(QLabel("找不到符合的圖片。試試圖片中的文字、檔名或更簡短的描述。"), 0, 0, 1, 4)
        for index, hit in enumerate(hits):
            self.results_layout.addWidget(ResultCard(hit), index // 4, index % 4)
        source = hits[0].rerank_source if hits else "deterministic"
        if source == "qwen3.5":
            self.search_status.setText("已完成 Qwen3.5 最終圖片審核。")
        elif source == "deterministic-fallback":
            self.search_status.setText("VLM 未能完成，本次已安全改用混合搜尋結果。")
        else:
            self.search_status.setText("本次為精確／條件查詢，已使用可解釋的混合搜尋結果。")

    @Slot(int, str)
    def on_search_failed(self, generation: int, error: str):
        if generation != self.search_generation:
            return
        self._clear_layout(self.results_layout)
        self.results_layout.addWidget(QLabel(f"搜尋失敗：{error}"), 0, 0, 1, 4)
        self.search_status.setText("搜尋發生錯誤，請到模型與效能頁查看狀態。")

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

        object_settings = QHBoxLayout()
        object_label = QLabel("物件模型")
        self.object_profile_choice = QComboBox()
        self.object_profile_choice.addItem("YOLOE-26S Prompt-Free（推薦）", "yoloe26s-pf")
        self.object_profile_choice.addItem("YOLOE-26M Prompt-Free（高準確）", "yoloe26m-pf")
        object_index = self.object_profile_choice.findData(self.config.active_object_profile)
        self.object_profile_choice.setCurrentIndex(max(0, object_index))
        apply_object = QPushButton("套用物件模型")
        apply_object.clicked.connect(self.apply_object_settings)
        object_settings.addWidget(object_label)
        object_settings.addWidget(self.object_profile_choice, 1)
        object_settings.addWidget(apply_object)
        self.models_layout.addLayout(object_settings)

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
        vocabulary_size = int(object_details.get("vocabulary_size", 0) or 0)
        object_name = str(object_details.get("scale", self.config.active_object_profile)).upper()
        cards.addWidget(StatusCard(
            "物件辨識",
            f"YOLOE-26 {object_name} Prompt-Free 已載入" if object_ready else "尚未安裝或載入失敗",
            f"固定開放詞彙：{vocabulary_size:,} · 實際裝置：{', '.join(object_devices) or '—'}",
            "ok" if any(str(item).startswith("GPU") for item in object_devices) else "warning",
        ), 2, 0)

        vlm_runtime = getattr(self.search_service.reranker.engine, "runtime_details", lambda: [])() if self.search_service.reranker else []
        vlm_installed = bool(vlm_runtime and (vlm_runtime[0].get("installed") or vlm_runtime[0].get("loaded")))
        vlm_devices = sorted({device for item in vlm_runtime for device in item.get("execution_devices", [])})
        cards.addWidget(StatusCard(
            "最終圖片審核",
            "Qwen3.5 已安裝" if vlm_installed else ("已停用" if self.config.vlm_policy == "off" else "尚未安裝"),
            f"模式：{self.config.vlm_policy} · 實際裝置：{', '.join(vlm_devices) or '等待首次使用'}",
            "ok" if vlm_installed else "warning",
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
        semantic_runtime = getattr(self.search_service.embeddings, "runtime_details", lambda: [])()
        for index, item in enumerate(ocr_runtime + semantic_runtime + object_runtime + vlm_runtime):
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

        locations = QLabel(f"模型位置：{self.paths.models}\n索引與設定：{self.paths.root}")
        locations.setObjectName("locationBox")
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

        buttons = QHBoxLayout()
        refresh = QPushButton("重新檢查狀態")
        refresh.clicked.connect(self.refresh_model_status)
        open_folder = QPushButton("開啟資料位置")
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.root))))
        buttons.addWidget(refresh)
        buttons.addWidget(open_folder)
        buttons.addStretch()
        self.models_layout.addLayout(buttons)

        if semantic_ready:
            self.capability_banner.setObjectName("capabilityOk")
            self.capability_banner.setText("✓ 圖片語意搜尋與文字搜尋都可以使用")
        else:
            self.capability_banner.setObjectName("capabilityWarning")
            self.capability_banner.setText("目前模式：可搜尋圖片中的文字與檔名。圖片語意模型尚未安裝，自然語言內容搜尋暫不可用。")
        self.capability_banner.style().unpolish(self.capability_banner)
        self.capability_banner.style().polish(self.capability_banner)

    def apply_object_settings(self):
        profile_id = str(self.object_profile_choice.currentData())
        manager = self.registry.object_manager(profile_id)
        local_ready = bool(
            list(manager.root.glob("*.xml"))
            and list(manager.root.glob("*.bin"))
            and (manager.root / "metadata.yaml").is_file()
        )
        if not local_ready and not manager.ready(verify_hashes=False):
            if manager.configured():
                if ModelSetupDialog(manager, self).exec() != QDialog.Accepted:
                    return
            else:
                QMessageBox.warning(
                    self,
                    "模型來源尚未設定",
                    "請先執行 YOLOE-26 WSL 轉換工具，或將 Hugging Face 固定版本寫入 models.json。",
                )
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

    def finish_setup(self, folder: str):
        self.config.setup_complete = True
        self.config.roots = [folder]
        self.config.save(self.paths.config)
        self.main.roots.clear()
        self.main.roots.addItems(self.config.roots)
        self.pages.setCurrentWidget(self.main)
        self.main.start_index(folder)


STYLES = """
QWidget { font-family: 'Segoe UI'; font-size: 14px; color: #172033; background: #f7f8fb; }
QLineEdit { background: white; border: 1px solid #d7dce5; border-radius: 9px; padding: 11px; }
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
QLabel#locationBox { background: #f2f4f7; border-radius: 8px; padding: 12px; color: #475467; }
QFrame#resultCard { background: white; border: 1px solid #e4e7ec; border-radius: 12px; min-width: 220px; max-width: 260px; }
QFrame#statusCard_ok, QFrame#statusCard_warning, QFrame#statusCard_error, QFrame#statusCard_neutral { background: white; border: 1px solid #e4e7ec; border-radius: 12px; min-height: 105px; }
QTabWidget::pane { border: 0; }
"""
