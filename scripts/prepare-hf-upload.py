"""Validate local OpenVINO packages and write a reproducible HF upload manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IGNORED_PARTS = {"compiled_cache", "__pycache__", ".git"}
BUNDLE_IDS = (
    "siglip2-so400m-patch16-naflex-openvino-fp16",
    "pp-ocrv6-medium-openvino-fp16",
    "yoloe-26s-pf-openvino-fp16",
    "yoloe-26m-pf-openvino-fp16",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_folder(folder: Path) -> str:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError(f"模型資料夾不存在或不安全：{folder}")
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"模型包不得包含符號連結：{path}")
    names = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if {"vision_model.xml", "text_model.xml", "position_embedding.npy"} <= names:
        return "siglip"
    if {"det_model.xml", "rec_model.xml", "rec_inference.yml"} <= names:
        return "ocr"
    if "metadata.yaml" in names and any(name.endswith(".xml") for name in names) and any(
        name.endswith(".bin") for name in names
    ):
        metadata = (folder / "metadata.yaml").read_text(encoding="utf-8", errors="replace").casefold()
        if "yoloe" in metadata or "prompt-free" in metadata or "prompt_free" in metadata:
            return "yoloe"
    if {"config.json", "processor_config.json", "openvino_language_model.xml"} <= names and any(
        name.startswith("openvino_vision") and name.endswith("_model.xml") for name in names
    ):
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") not in {"qwen3_5", "qwen3_vl"}:
            raise ValueError("只接受已測試的 qwen3_5 或 qwen3_vl OpenVINO VLM")
        return "vlm"
    raise ValueError("無法辨識或模型包缺少必要 OpenVINO IR")


def model_files(folder: Path, root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.name in {"upload-manifest.json"}:
            continue
        files.append({
            "name": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", help="預計上傳的 Hugging Face repo，例如 owner/visual-search-openvino-models")
    args = parser.parse_args()
    folder = args.folder.resolve()

    bundle_folders = [folder / model_id for model_id in BUNDLE_IDS]
    is_bundle = any(path.exists() for path in bundle_folders)
    if is_bundle:
        missing = [path.name for path in bundle_folders if not path.is_dir()]
        if missing:
            raise ValueError("整合模型庫缺少資料夾：" + ", ".join(missing))
        kinds = {path.name: validate_folder(path) for path in bundle_folders}
        files = model_files(folder, folder)
        kind: object = "visual-search-openvino-bundle"
    else:
        kinds = {}
        kind = validate_folder(folder)
        files = model_files(folder, folder)

    if not files:
        raise ValueError("模型包沒有可上傳檔案")
    payload = {
        "schema_version": 2,
        "kind": kind,
        "models": kinds,
        "repo": args.repo or "",
        "total_size": sum(int(item["size"]) for item in files),
        "files": files,
    }
    output = folder / "upload-manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {kind}: {len(files)} files, {payload['total_size'] / 1e9:.2f} GB")
    print(f"Upload manifest: {output}")
    if args.repo:
        print(f"Publish command: python scripts/publish-model.py --bundle --repo {args.repo}")


if __name__ == "__main__":
    main()
