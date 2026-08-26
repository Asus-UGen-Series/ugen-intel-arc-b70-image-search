from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import shutil
import tempfile
import threading
import zipfile
import stat
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


MODEL_ID = "siglip2-so400m-patch16-naflex-openvino-fp16"
MODEL_DIRECTORY = MODEL_ID
OCR_MODEL_ID = "pp-ocrv6-medium-openvino-fp16"
OCR_MODEL_DIRECTORY = OCR_MODEL_ID
YOLOE_S_ID = "yoloe-26s-pf-openvino-fp16"
YOLOE_M_ID = "yoloe-26m-pf-openvino-fp16"
# Compatibility aliases for older integrations. New code selects an object
# profile through ModelRegistry instead of assuming one fixed YOLO package.
YOLO_MODEL_ID = YOLOE_S_ID
YOLO_MODEL_DIRECTORY = YOLOE_S_ID
VLM_INT8_ID = "qwen3.5-9b-int8-openvino"
VLM_INT4_ID = "qwen3.5-9b-int4-openvino"

SECTION_DEFAULTS = {
    "model": MODEL_ID,
    "ocr_model": OCR_MODEL_ID,
    "yoloe_s": YOLOE_S_ID,
    "yoloe_m": YOLOE_M_ID,
    "vlm_int8": VLM_INT8_ID,
    "vlm_int4": VLM_INT4_ID,
}


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int = 0


class ModelManager:
    """Downloads one immutable, checksummed OpenVINO model package.

    The shipped manifest is deliberately empty during development. The publish
    script fills it with URLs pinned to a Hugging Face commit, so end users never
    download a moving ``main`` revision and never need a Hugging Face account.
    """

    def __init__(self, root: Path, manifest: Path, section: str = "model"):
        self.root = root
        self.manifest = manifest
        self._data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.section = section

    @property
    def entry(self) -> dict[str, object]:
        return dict(self._data.get(self.section, {}))

    @property
    def model_id(self) -> str:
        return str(self.entry.get("id", SECTION_DEFAULTS.get(self.section, self.section)))

    @property
    def source_repo(self) -> str:
        return str(self.entry.get("source_repo", ""))

    @property
    def revision(self) -> str:
        return str(self.entry.get("revision", ""))

    def files(self) -> list[ModelFile]:
        return [ModelFile(**item) for item in self.entry.get("files", [])]

    def configured(self) -> bool:
        files = self.files()
        return bool(self.source_repo and self.revision and files and all(item.url and item.sha256 for item in files))

    @staticmethod
    def checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def total_size(self) -> int:
        return sum(item.size for item in self.files())

    @property
    def verification_marker(self) -> Path:
        return self.root / ".verified.json"

    def downloaded(self, verify_hashes: bool = True) -> bool:
        files = self.files()
        if not files:
            return False
        for item in files:
            destination = self.root / item.name
            if not destination.exists():
                return False
            if item.size and destination.stat().st_size != item.size:
                return False
            if verify_hashes and self.checksum(destination) != item.sha256:
                return False
        return True

    def ready(self, verify_hashes: bool = True) -> bool:
        if not self.downloaded(verify_hashes=verify_hashes) or not self.verification_marker.exists():
            return False
        try:
            marker = json.loads(self.verification_marker.read_text(encoding="utf-8"))
            return marker.get("revision") == self.revision and marker.get("model_id") == self.model_id
        except (OSError, ValueError):
            return False

    def mark_verified(self, device: str) -> None:
        payload = {"model_id": self.model_id, "revision": self.revision, "device": device}
        temporary = self.verification_marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.verification_marker)

    def download(self, progress: Callable[[str, int, int], None] | None = None) -> None:
        if not self.configured():
            raise RuntimeError("發行模型下載來源尚未設定，請先執行 scripts/publish-model.py。")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.source_repo and all("huggingface.co/" in item.url for item in self.files()):
            self._download_huggingface(progress)
            return
        for item in self.files():
            destination = self.root / item.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and self.checksum(destination) == item.sha256:
                if progress:
                    progress(item.name, item.size, item.size)
                continue
            partial = destination.with_suffix(destination.suffix + ".part")
            for attempt in range(3):
                try:
                    existing = partial.stat().st_size if partial.exists() else 0
                    request = urllib.request.Request(item.url, headers={"User-Agent": "VisualSearch/1.0"})
                    if existing:
                        request.add_header("Range", f"bytes={existing}-")
                    with urllib.request.urlopen(request, timeout=45) as response:
                        # Some CDNs ignore Range. Starting from zero avoids appending
                        # a complete response to a partial file.
                        resumed = existing and getattr(response, "status", 200) == 206
                        mode = "ab" if resumed else "wb"
                        received = existing if resumed else 0
                        total = item.size or int(response.headers.get("Content-Length", 0)) + received
                        with partial.open(mode) as output:
                            while chunk := response.read(1024 * 1024):
                                output.write(chunk)
                                received += len(chunk)
                                if progress:
                                    progress(item.name, received, total)
                    if item.size and partial.stat().st_size != item.size:
                        raise ValueError(f"File size mismatch: {item.name}")
                    if self.checksum(partial) != item.sha256:
                        raise ValueError(f"SHA-256 mismatch: {item.name}")
                    partial.replace(destination)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)

    def _download_huggingface(self, progress: Callable[[str, int, int], None] | None) -> None:
        """Use the official resumable Xet/LFS client for immutable HF packages."""
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        callback_lock = threading.Lock()
        known = {item.name: item for item in self.files()}

        class DownloadProgress(tqdm):
            def __init__(bar, *args, **kwargs):
                bar.file_name = str(kwargs.get("desc") or "model")
                super().__init__(*args, **kwargs)

            def update(bar, amount=1):
                result = super().update(amount)
                if progress and bar.total:
                    with callback_lock:
                        progress(bar.file_name, int(bar.n), int(bar.total))
                return result

        snapshot_download(
            self.source_repo,
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=list(known),
            max_workers=4,
            tqdm_class=DownloadProgress,
            user_agent="VisualSearch/1.0",
        )
        for item in self.files():
            destination = self.root / item.name
            if not destination.is_file():
                raise FileNotFoundError(f"Downloaded model is missing: {item.name}")
            if item.size and destination.stat().st_size != item.size:
                raise ValueError(f"File size mismatch: {item.name}")
            if self.checksum(destination) != item.sha256:
                raise ValueError(f"SHA-256 mismatch: {item.name}")
            if progress:
                progress(item.name, item.size, item.size)
            destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    section: str
    display_name: str
    precision: str
    family: str
    expected_size: int
    recommended: bool = False


class ModelRegistry:
    """Catalogues immutable factory models and safely managed local models."""

    VLM_PROFILES = {
        "qwen35-9b-int8": ModelProfile(
            "qwen35-9b-int8", "vlm_int8", "Qwen3.5 9B 高品質", "INT8", "qwen3_5",
            9_460_000_000, True,
        ),
        "qwen35-9b-int4": ModelProfile(
            "qwen35-9b-int4", "vlm_int4", "Qwen3.5 9B 快速", "INT4", "qwen3_5",
            6_130_000_000, False,
        ),
    }
    OBJECT_PROFILES = {
        "yoloe26s-pf": ModelProfile(
            "yoloe26s-pf", "yoloe_s", "YOLOE-26S Prompt-Free", "FP16", "yoloe26_pf",
            45_000_000, True,
        ),
        "yoloe26m-pf": ModelProfile(
            "yoloe26m-pf", "yoloe_m", "YOLOE-26M Prompt-Free", "FP16", "yoloe26_pf",
            95_000_000, False,
        ),
    }
    ALLOWED_VLM_FAMILIES = {"qwen3_5", "qwen3_vl"}

    def __init__(self, models_root: Path, manifest: Path):
        self.models_root = Path(models_root)
        self.manifest = Path(manifest)
        self.models_root.mkdir(parents=True, exist_ok=True)

    def manager(self, section: str) -> ModelManager:
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        entry = data.get(section, {})
        directory = str(entry.get("directory") or SECTION_DEFAULTS.get(section, section))
        return ModelManager(self.models_root / directory, self.manifest, section)

    def vlm_profile(self, profile_id: str) -> ModelProfile:
        if profile_id not in self.VLM_PROFILES:
            raise KeyError(f"Unknown VLM profile: {profile_id}")
        return self.VLM_PROFILES[profile_id]

    def vlm_manager(self, profile_id: str) -> ModelManager:
        return self.manager(self.vlm_profile(profile_id).section)

    def object_profile(self, profile_id: str) -> ModelProfile:
        if profile_id not in self.OBJECT_PROFILES:
            raise KeyError(f"Unknown object profile: {profile_id}")
        return self.OBJECT_PROFILES[profile_id]

    def object_manager(self, profile_id: str) -> ModelManager:
        return self.manager(self.object_profile(profile_id).section)

    @staticmethod
    def required_free_space(profile: ModelProfile) -> int:
        return int(profile.expected_size * 1.2) + 2 * 1024 ** 3

    @staticmethod
    def _safe_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts or Path(name).drive:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symbolic links are not allowed: {member.filename}")
            yield member

    @staticmethod
    def _reject_links(folder: Path) -> None:
        if folder.is_symlink():
            raise ValueError("Symbolic-link model folders are not allowed")
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symbolic links are not allowed in local models: {path}")

    @classmethod
    def validate_vlm_directory(cls, folder: Path) -> dict[str, object]:
        folder = Path(folder)
        config_path = folder / "config.json"
        if not config_path.is_file():
            raise ValueError("缺少 config.json；請選擇完整的 OpenVINO VLM 資料夾")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        family = str(config.get("model_type", ""))
        if family not in cls.ALLOWED_VLM_FAMILIES:
            raise ValueError(f"不支援的 VLM family：{family or 'unknown'}")
        required = [
            "openvino_language_model.xml", "openvino_language_model.bin",
            "tokenizer_config.json", "processor_config.json",
        ]
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise ValueError("OpenVINO VLM 不完整：" + ", ".join(missing))
        vision_xml = list(folder.glob("openvino_vision*_model.xml"))
        if not vision_xml:
            raise ValueError("缺少 OpenVINO vision model")
        return {"family": family, "required": required, "vision_models": [item.name for item in vision_xml]}

    @classmethod
    def validate_object_directory(cls, folder: Path, profile_id: str) -> dict[str, object]:
        """Validate a converted YOLOE-26 prompt-free OpenVINO package."""
        folder = Path(folder)
        cls._reject_links(folder)
        metadata_path = folder / "metadata.yaml"
        if not metadata_path.is_file():
            raise ValueError("缺少 metadata.yaml；請選擇完整的 YOLOE-26 OpenVINO 資料夾")
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        xml_models = sorted(folder.glob("*.xml"))
        paired_models = [path for path in xml_models if path.with_suffix(".bin").is_file()]
        if not paired_models:
            raise ValueError("找不到成對的 OpenVINO .xml 與 .bin 模型檔")
        prompt_mode = str(metadata.get("prompt_mode", "")).casefold().replace("_", "-")
        family = str(metadata.get("visual_search_family", "")).casefold()
        if family and family != "yoloe26_pf":
            raise ValueError(f"模型 family 不符：{family}")
        if prompt_mode and prompt_mode != "prompt-free":
            raise ValueError("只接受 YOLOE-26 Prompt-Free 模型")
        expected_scale = "S" if profile_id == "yoloe26s-pf" else "M"
        actual_scale = str(metadata.get("scale", "")).upper()
        if actual_scale and actual_scale != expected_scale:
            raise ValueError(f"選擇的是 YOLOE-26{expected_scale}，但匯入模型標示為 {actual_scale}")
        return {
            "profile_id": profile_id,
            "scale": actual_scale or expected_scale,
            "model": paired_models[0].name,
            "prompt_mode": prompt_mode or "prompt-free",
        }

    def import_local_vlm(self, source: Path) -> Path:
        source = Path(source)
        staging_parent = self.models_root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="vlm-", dir=staging_parent))
        try:
            if source.is_file() and source.suffix.casefold() == ".zip":
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(staging, members=self._safe_members(archive))
                candidates = [staging, *[item for item in staging.iterdir() if item.is_dir()]]
                model_dir = next((item for item in candidates if (item / "config.json").is_file()), staging)
            elif source.is_dir():
                self._reject_links(source)
                model_dir = source
            else:
                raise ValueError("請選擇 OpenVINO 模型資料夾或 ZIP")
            details = self.validate_vlm_directory(model_dir)
            fingerprint = self.fingerprint(model_dir)[:16]
            target = self.models_root / "custom" / f"{details['family']}-{fingerprint}"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".new")
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.copytree(model_dir, temporary, ignore=shutil.ignore_patterns("compiled_cache"))
            if target.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            else:
                temporary.replace(target)
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def import_local_object(self, source: Path, profile_id: str) -> Path:
        """Safely import a folder/ZIP into the selected factory object-model slot."""
        source = Path(source)
        manager = self.object_manager(profile_id)
        staging_parent = self.models_root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="yoloe-", dir=staging_parent))
        try:
            if source.is_file() and source.suffix.casefold() == ".zip":
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(staging, members=self._safe_members(archive))
                candidates = [staging, *[item for item in staging.rglob("*") if item.is_dir()]]
                model_dir = next((item for item in candidates if (item / "metadata.yaml").is_file()), staging)
            elif source.is_dir():
                self._reject_links(source)
                model_dir = source
            else:
                raise ValueError("請選擇 YOLOE-26 OpenVINO 模型資料夾或 ZIP")
            self.validate_object_directory(model_dir, profile_id)
            target = manager.root
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".new")
            backup = target.with_name(target.name + ".previous")
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
            shutil.copytree(model_dir, temporary, ignore=shutil.ignore_patterns("compiled_cache"))
            self.validate_object_directory(temporary, profile_id)
            if target.exists():
                target.replace(backup)
            try:
                temporary.replace(target)
            except Exception:
                if backup.exists() and not target.exists():
                    backup.replace(target)
                raise
            shutil.rmtree(backup, ignore_errors=True)
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def fingerprint(folder: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in Path(folder).rglob("*") if item.is_file() and "compiled_cache" not in item.parts):
            digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            if path.suffix.casefold() in {".json", ".xml", ".jinja"}:
                digest.update(ModelManager.checksum(path).encode("ascii"))
        return digest.hexdigest()
