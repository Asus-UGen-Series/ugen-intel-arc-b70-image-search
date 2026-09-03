"""Publish tested OpenVINO packages and pin immutable download metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi, create_repo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "src" / "visual_search" / "resources" / "models.json"
BUNDLE_ROOT = PROJECT_ROOT / "release-models" / "visual-search-openvino-models"

CUSTOM_SECTIONS = {
    "model": {
        "id": "siglip2-so400m-patch16-naflex-openvino-fp16",
        "required": ["vision_model.xml", "vision_model.bin", "text_model.xml", "text_model.bin", "position_embedding.npy"],
        "upstream": "google/siglip2-so400m-patch16-naflex",
    },
    "ocr_model": {
        "id": "pp-ocrv6-medium-openvino-fp16",
        "required": ["det_model.xml", "det_model.bin", "rec_model.xml", "rec_model.bin", "rec_inference.yml", "openvino_config.json"],
        "upstream": ["PaddlePaddle/PP-OCRv6_medium_det_onnx", "PaddlePaddle/PP-OCRv6_medium_rec_onnx"],
    },
    "yoloe_s": {
        "id": "yoloe-26s-pf-openvino-fp16",
        "required": ["metadata.yaml", "vocabulary.json"],
        "upstream": "Ultralytics/YOLOE-26S-PF",
    },
    "yoloe_m": {
        "id": "yoloe-26m-pf-openvino-fp16",
        "required": ["metadata.yaml", "vocabulary.json"],
        "upstream": "Ultralytics/YOLOE-26M-PF",
    },
}

OFFICIAL_COMPAT_SECTIONS = {
    "vlm_int8": {
        "id": "qwen3.5-9b-int8-openvino",
        "folder": PROJECT_ROOT / "artifacts" / "hf" / "Qwen3.5-9B-int8-ov",
        "required": ["config.json", "processor_config.json", "openvino_language_model.xml", "openvino_language_model.bin"],
        "upstream": "Qwen/Qwen3.5-9B",
    },
    "vlm_int4": {
        "id": "qwen3.5-9b-int4-openvino",
        "folder": PROJECT_ROOT / "artifacts" / "hf" / "Qwen3.5-9B-int4-ov",
        "required": ["config.json", "processor_config.json", "openvino_language_model.xml", "openvino_language_model.bin"],
        "upstream": "Qwen/Qwen3.5-9B",
    },
}
ALL_SECTIONS = {**CUSTOM_SECTIONS, **OFFICIAL_COMPAT_SECTIONS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_folder(section: str) -> Path:
    if section in CUSTOM_SECTIONS:
        return BUNDLE_ROOT / str(CUSTOM_SECTIONS[section]["id"])
    return Path(OFFICIAL_COMPAT_SECTIONS[section]["folder"])


def validate_package(section: str, folder: Path) -> None:
    specification = ALL_SECTIONS[section]
    missing = [name for name in specification["required"] if not (folder / name).is_file()]
    if section.startswith("yoloe_"):
        if not list(folder.glob("*.xml")):
            missing.append("<OpenVINO .xml>")
        if not list(folder.glob("*.bin")):
            missing.append("<OpenVINO .bin>")
    if missing:
        raise SystemExit(f"{section} model package incomplete: {', '.join(missing)}")


def package_files(folder: Path, repo_path: str, repo: str, revision: str) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        relative = path.relative_to(folder).as_posix()
        if "compiled_cache" in path.parts or relative == "upload-manifest.json":
            continue
        source_name = f"{repo_path}/{relative}" if repo_path else relative
        files.append({
            "name": relative,
            "source_name": source_name,
            "url": f"https://huggingface.co/{repo}/resolve/{revision}/{quote(source_name)}?download=true",
            "sha256": sha256(path),
            "size": path.stat().st_size,
        })
    return files


def update_manifest(
    manifest_path: Path,
    sections: list[str],
    folders: dict[str, Path],
    repo: str,
    revision: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_version"] = 5
    for section in sections:
        specification = ALL_SECTIONS[section]
        repo_path = str(specification["id"]) if section in CUSTOM_SECTIONS else ""
        entry = dict(manifest.get(section, {}))
        entry.update({
            "id": specification["id"],
            "directory": specification["id"],
            "source_repo": repo,
            "revision": revision,
            "upstream": specification["upstream"],
            "files": package_files(folders[section], repo_path, repo, revision),
        })
        manifest[section] = entry
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="例如：使用者名稱/visual-search-openvino-models")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--bundle", action="store_true", help="一次發布四套自製模型至同一個 repo")
    target.add_argument("--section", choices=ALL_SECTIONS, help="相容的單一模型發布模式")
    parser.add_argument("--folder", type=Path, help="單一模型模式的來源資料夾")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    sections = list(CUSTOM_SECTIONS) if args.bundle else [str(args.section)]
    folders = {
        section: (args.folder.resolve() if args.folder and len(sections) == 1 else default_folder(section).resolve())
        for section in sections
    }
    for section, folder in folders.items():
        validate_package(section, folder)

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    if args.bundle:
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=BUNDLE_ROOT,
            commit_message="Publish tested Visual Search OpenVINO model bundle",
            ignore_patterns=["**/compiled_cache/**", "**/upload-manifest.json"],
        )
    else:
        section = sections[0]
        repo_path = str(ALL_SECTIONS[section]["id"]) if section in CUSTOM_SECTIONS else "."
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=folders[section],
            path_in_repo=repo_path,
            commit_message=f"Publish tested {section} OpenVINO package",
            ignore_patterns=["compiled_cache/**", "upload-manifest.json"],
        )

    revision = api.model_info(args.repo, revision="main").sha
    update_manifest(args.manifest.resolve(), sections, folders, args.repo, revision)
    print(f"Published: https://huggingface.co/{args.repo}")
    print(f"Pinned revision: {revision}")
    print("Updated sections: " + ", ".join(sections))


if __name__ == "__main__":
    main()
