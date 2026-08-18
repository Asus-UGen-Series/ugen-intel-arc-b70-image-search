from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import zipfile
from importlib.resources import files
from pathlib import Path

from .config import AppConfig
from .engines import ColorFallbackEmbedding, NoopOcrEngine, OpenVinoSiglipEngine, detect_openvino_devices
from .indexer import IndexController, SearchService
from .models import (
    MODEL_DIRECTORY, OCR_MODEL_DIRECTORY,
    ModelManager, ModelRegistry,
)
from .ocr_openvino import OpenVinoPaddleOcrEngine
from .paths import AppPaths
from .storage import SQLiteAssetRepository
from .vectors import USearchVectorIndex
from .gpu_scheduler import GpuScheduler
from .vlm import LazyVlmEngine, NoopVlmEngine, VlmReranker
from .yolo_openvino import OpenVinoYoloEngine


def configure_logging(paths: AppPaths) -> None:
    paths.ensure()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=[logging.FileHandler(paths.logs / "visual-search.log", encoding="utf-8"), logging.StreamHandler()])


def manifest_path() -> Path:
    override = os.environ.get("VISUAL_SEARCH_MODEL_MANIFEST")
    return Path(override) if override else Path(str(files("visual_search").joinpath("resources/models.json")))


def create_model_registry(paths: AppPaths) -> ModelRegistry:
    return ModelRegistry(paths.models, manifest_path())


def local_object_package_ready(folder: Path) -> bool:
    """Recognise a converted YOLOE package before it has a release marker."""
    xml_files = list(folder.glob("*.xml"))
    return bool(
        xml_files
        and any(item.with_suffix(".bin").is_file() for item in xml_files)
        and (folder / "metadata.yaml").is_file()
    )


def create_services(paths: AppPaths, config: AppConfig | None = None, registry: ModelRegistry | None = None):
    config = config or AppConfig.load(paths.config)
    registry = registry or create_model_registry(paths)
    repository = SQLiteAssetRepository(paths.database)
    model_dir = paths.models / MODEL_DIRECTORY
    manager = registry.manager("model")
    local_model_ready = all((model_dir / name).is_file() for name in (
        "vision_model.xml", "vision_model.bin", "text_model.xml", "text_model.bin", "position_embedding.npy",
    ))
    model_ready = manager.ready(verify_hashes=False) or local_model_ready
    try:
        embeddings = OpenVinoSiglipEngine(model_dir) if model_ready else ColorFallbackEmbedding()
    except Exception:
        logging.exception("Semantic model unavailable; using filename/OCR search")
        embeddings = ColorFallbackEmbedding()
    ocr_manager = registry.manager("ocr_model")
    ocr_model_dir = paths.models / OCR_MODEL_DIRECTORY
    local_ocr_ready = all((ocr_model_dir / name).is_file() for name in (
        "det_model.xml", "det_model.bin", "rec_model.xml", "rec_model.bin", "rec_inference.yml",
    ))
    try:
        if ocr_manager.ready(verify_hashes=False) or local_ocr_ready:
            try:
                ocr = OpenVinoPaddleOcrEngine(ocr_model_dir, device="GPU")
            except Exception:
                logging.exception("Intel GPU OCR compilation failed; using OpenVINO CPU compatibility mode")
                ocr = OpenVinoPaddleOcrEngine(ocr_model_dir, device="CPU")
        else:
            logging.warning("OpenVINO OCR model unavailable; OCR is disabled until the model is installed")
            ocr = NoopOcrEngine()
    except Exception:
        logging.exception("OCR runtime unavailable; OCR is disabled")
        ocr = NoopOcrEngine()
    object_manager = registry.object_manager(config.active_object_profile)
    object_dir = object_manager.root
    try:
        object_ready = object_manager.ready(verify_hashes=False) or local_object_package_ready(object_dir)
        objects = OpenVinoYoloEngine(object_dir, "GPU") if object_ready else None
    except Exception:
        logging.exception("YOLOE OpenVINO unavailable; object search is disabled")
        objects = None
    scheduler = GpuScheduler()
    vectors = USearchVectorIndex(paths.vectors.with_name(f"vectors-{embeddings.dimensions}.usearch"), embeddings.dimensions)
    region_vectors = None
    if isinstance(embeddings, OpenVinoSiglipEngine):
        try:
            region_vectors = USearchVectorIndex(
                paths.vectors.with_name(f"regions-{embeddings.dimensions}.usearch"),
                embeddings.dimensions,
                dtype="f16",
            )
        except Exception:
            logging.exception("Region vector index unavailable; global semantic search remains active")
    vlm = NoopVlmEngine()
    if config.vlm_policy != "off":
        try:
            if config.active_vlm_profile == "local" and config.vlm_local_path:
                local_vlm = Path(config.vlm_local_path)
                registry.validate_vlm_directory(local_vlm)
                vlm = LazyVlmEngine(local_vlm, "GPU", config.vlm_image_size, "本機 VLM")
            elif config.active_vlm_profile in registry.VLM_PROFILES:
                profile = registry.vlm_profile(config.active_vlm_profile)
                vlm_manager = registry.vlm_manager(config.active_vlm_profile)
                if vlm_manager.ready(verify_hashes=False):
                    vlm = LazyVlmEngine(vlm_manager.root, "GPU", config.vlm_image_size,
                                        f"{profile.display_name} {profile.precision}")
        except Exception:
            logging.exception("VLM unavailable; deterministic search remains active")
            vlm = NoopVlmEngine()
    reranker = VlmReranker(
        repository, vlm, scheduler, config.vlm_policy,
        config.vlm_candidate_count, config.vlm_timeout_seconds,
    )
    controller = IndexController(
        repository, vectors, ocr, embeddings, paths.thumbnails,
        objects, scheduler, region_vectors,
    )
    return repository, controller, SearchService(
        repository, vectors, embeddings, scheduler, reranker, region_vectors,
    )


def create_model_manager(paths: AppPaths, section: str = "model") -> ModelManager:
    return create_model_registry(paths).manager(section)


def diagnose(paths: AppPaths) -> int:
    config = AppConfig.load(paths.config)
    registry = create_model_registry(paths)
    repository, controller, search = create_services(paths, config, registry)
    manager = create_model_manager(paths)
    model_dir = paths.models / MODEL_DIRECTORY
    local_model_ready = all((model_dir / name).is_file() for name in (
        "vision_model.xml", "vision_model.bin", "text_model.xml", "text_model.bin", "position_embedding.npy",
    ))
    ocr_manager = create_model_manager(paths, "ocr_model")
    ocr_model_dir = paths.models / OCR_MODEL_DIRECTORY
    local_ocr_ready = all((ocr_model_dir / name).is_file() for name in (
        "det_model.xml", "det_model.bin", "rec_model.xml", "rec_model.bin", "rec_inference.yml",
    ))
    object_manager = registry.object_manager(config.active_object_profile)
    object_local_ready = local_object_package_ready(object_manager.root)
    vlm_manager = registry.vlm_manager(config.active_vlm_profile) if config.active_vlm_profile in registry.VLM_PROFILES else None
    report = {
        "application_data": str(paths.root),
        "database": str(paths.database),
        "indexed_images": repository.count(),
        "index_status": repository.counts_by_status(),
        "openvino": detect_openvino_devices(),
        "semantic_model": {
            "id": manager.model_id,
            "download_configured": manager.configured(),
            "files_ready": manager.ready(verify_hashes=False) or local_model_ready,
            "local_development_package": local_model_ready and not manager.configured(),
            "source_repo": manager.source_repo,
            "revision": manager.revision,
        },
        "ocr_model": {
            "id": ocr_manager.model_id,
            "download_configured": ocr_manager.configured(),
            "files_ready": ocr_manager.ready(verify_hashes=False) or local_ocr_ready,
            "local_development_package": local_ocr_ready and not ocr_manager.configured(),
            "source_repo": ocr_manager.source_repo,
            "revision": ocr_manager.revision,
        },
        "object_model": {
            "profile": config.active_object_profile,
            "id": object_manager.model_id,
            "download_configured": object_manager.configured(),
            "files_ready": object_manager.ready(verify_hashes=False) or object_local_ready,
            "local_development_package": object_local_ready and not object_manager.configured(),
            "source_repo": object_manager.source_repo,
            "revision": object_manager.revision,
        },
        "region_index": {
            "enabled": controller.region_vectors is not None,
            "vectors": repository.region_count(),
            "storage": str(paths.vectors.with_name(f"regions-{search.embeddings.dimensions}.usearch")),
        },
        "vlm_model": {
            "profile": config.active_vlm_profile, "policy": config.vlm_policy,
            "files_ready": bool(vlm_manager and vlm_manager.ready(verify_hashes=False)) or bool(config.vlm_local_path),
            "source_repo": vlm_manager.source_repo if vlm_manager else "local",
            "revision": vlm_manager.revision if vlm_manager else "",
        },
        "gpu_scheduler": controller.scheduler.snapshot() if controller.scheduler else {},
        "model_execution": (
            runtime_details(controller.ocr) + runtime_details(search.embeddings)
            + runtime_details(controller.objects) + runtime_details(search.reranker.engine)
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    (paths.logs / "latest-diagnostics.json").write_text(rendered + "\n", encoding="utf-8")
    cli_output(rendered)
    return 0


def runtime_details(engine) -> list[dict[str, object]]:
    method = getattr(engine, "runtime_details", None)
    return method() if callable(method) else []


def cli_output(value: str) -> None:
    """Windowed PyInstaller builds have no stdout; CLI commands must not fail."""
    if getattr(sys, "stdout", None) is not None:
        print(value)


def reset(paths: AppPaths) -> int:
    for target in (paths.database, paths.database.with_suffix(".db-wal"), paths.database.with_suffix(".db-shm")):
        target.unlink(missing_ok=True)
    for target in paths.root.glob("vectors-*.usearch"):
        target.unlink(missing_ok=True)
    for target in paths.root.glob("regions-*.usearch"):
        target.unlink(missing_ok=True)
    shutil.rmtree(paths.thumbnails, ignore_errors=True)
    paths.ensure()
    cli_output("索引已重設；原始圖片沒有被刪除。")
    return 0


def export_logs(paths: AppPaths, destination: Path | None) -> int:
    destination = destination or Path.cwd() / "VisualSearch-diagnostics.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for log in paths.logs.glob("*.log"):
            archive.write(log, f"logs/{log.name}")
        if paths.config.exists():
            archive.write(paths.config, "config.json")
    cli_output(str(destination))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--reset-index", action="store_true")
    parser.add_argument("--export-logs", nargs="?", const="", metavar="ZIP")
    args = parser.parse_args(argv)
    paths = AppPaths.default()
    configure_logging(paths)
    if args.diagnose:
        return diagnose(paths)
    if args.reset_index:
        return reset(paths)
    if args.export_logs is not None:
        return export_logs(paths, Path(args.export_logs) if args.export_logs else None)
    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow, ModelSetupDialog, VlmChoiceDialog
    app = QApplication(sys.argv)
    app.setApplicationName("Visual Search")
    config = AppConfig.load(paths.config)
    registry = create_model_registry(paths)
    manager = registry.manager("model")
    ocr_manager = registry.manager("ocr_model")
    object_manager = registry.object_manager(config.active_object_profile)
    # SHA-256 is verified while downloading. Normal launches only check pinned
    # file sizes so a multi-GB model does not get re-hashed on every startup.
    if manager.configured() and not manager.ready(verify_hashes=False):
        ModelSetupDialog(manager).exec()
    if ocr_manager.configured() and not ocr_manager.ready(verify_hashes=False):
        ModelSetupDialog(ocr_manager).exec()
    if object_manager.configured() and not object_manager.ready(verify_hashes=False):
        ModelSetupDialog(object_manager).exec()
    if not config.vlm_setup_complete:
        VlmChoiceDialog(registry, config, paths).exec()
    elif config.vlm_policy != "off" and config.active_vlm_profile in registry.VLM_PROFILES:
        selected_vlm = registry.vlm_manager(config.active_vlm_profile)
        if selected_vlm.configured() and not selected_vlm.ready(verify_hashes=False):
            ModelSetupDialog(selected_vlm).exec()
    config = AppConfig.load(paths.config)
    repository, controller, search = create_services(paths, config, registry)
    window = MainWindow(paths, config, controller, search, manager, registry)
    if controller.scheduler is not None:
        app.aboutToQuit.connect(lambda: controller.scheduler.shutdown(wait=False))
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
