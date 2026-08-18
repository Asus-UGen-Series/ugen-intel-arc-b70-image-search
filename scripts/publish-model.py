"""Upload a tested OpenVINO package and update one immutable app manifest section."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi, create_repo


DEFAULT_MANIFEST = Path("src/visual_search/resources/models.json")
SECTIONS = {
    "model": {
        "id": "siglip2-so400m-patch16-naflex-openvino-fp16",
        "folder": Path("release-models/siglip2-so400m-patch16-naflex-openvino-fp16"),
        "required": ["vision_model.xml", "vision_model.bin", "text_model.xml", "text_model.bin", "position_embedding.npy"],
        "upstream": "google/siglip2-so400m-patch16-naflex",
    },
    "ocr_model": {
        "id": "pp-ocrv6-medium-openvino-fp16",
        "folder": Path("release-models/pp-ocrv6-medium-openvino-fp16"),
        "required": ["det_model.xml", "det_model.bin", "rec_model.xml", "rec_model.bin", "rec_inference.yml", "openvino_config.json"],
        "upstream": ["PaddlePaddle/PP-OCRv6_medium_det_onnx", "PaddlePaddle/PP-OCRv6_medium_rec_onnx"],
    },
    "yoloe_s": {
        "id": "yoloe-26s-pf-openvino-fp16",
        "folder": Path("artifacts/hf/yoloe-26s-pf-openvino-fp16"),
        "required": ["metadata.yaml", "vocabulary.json"],
        "upstream": "Ultralytics/YOLOE-26S-PF",
    },
    "yoloe_m": {
        "id": "yoloe-26m-pf-openvino-fp16",
        "folder": Path("artifacts/hf/yoloe-26m-pf-openvino-fp16"),
        "required": ["metadata.yaml", "vocabulary.json"],
        "upstream": "Ultralytics/YOLOE-26M-PF",
    },
    "vlm_int8": {
        "id": "qwen3.5-9b-int8-openvino",
        "folder": Path("artifacts/hf/Qwen3.5-9B-int8-ov"),
        "required": ["config.json", "processor_config.json", "openvino_language_model.xml", "openvino_language_model.bin"],
        "upstream": "Qwen/Qwen3.5-9B",
    },
    "vlm_int4": {
        "id": "qwen3.5-9b-int4-openvino",
        "folder": Path("artifacts/hf/Qwen3.5-9B-int4-ov"),
        "required": ["config.json", "processor_config.json", "openvino_language_model.xml", "openvino_language_model.bin"],
        "upstream": "Qwen/Qwen3.5-9B",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="For example: your-name/visual-search-ppocrv6-openvino")
    parser.add_argument("--section", choices=SECTIONS, default="model")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    specification = SECTIONS[args.section]
    folder = (args.folder or specification["folder"]).resolve()
    missing = [name for name in specification["required"] if not (folder / name).is_file()]
    if args.section.startswith("yoloe_"):
        if not list(folder.glob("*.xml")):
            missing.append("<OpenVINO .xml>")
        if not list(folder.glob("*.bin")):
            missing.append("<OpenVINO .bin>")
    if missing:
        raise SystemExit(f"Model package incomplete: {', '.join(missing)}")

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=folder,
        commit_message="Publish tested OpenVINO FP16 package",
        ignore_patterns=["compiled_cache/**"],
    )
    revision = api.model_info(args.repo, revision="main").sha
    files = []
    for path in sorted(item for item in folder.rglob("*") if item.is_file() and "compiled_cache" not in item.parts):
        name = path.relative_to(folder).as_posix()
        files.append({
            "name": name,
            "url": f"https://huggingface.co/{args.repo}/resolve/{revision}/{quote(name)}?download=true",
            "sha256": sha256(path),
            "size": path.stat().st_size,
        })

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {"manifest_version": 5}
    manifest["manifest_version"] = 5
    entry = dict(manifest.get(args.section, {}))
    entry.update({
        "id": specification["id"],
        "directory": specification["id"],
        "source_repo": args.repo,
        "revision": revision,
        "upstream": specification["upstream"],
        "files": files,
    })
    manifest[args.section] = entry
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Published: https://huggingface.co/{args.repo}")
    print(f"Pinned revision: {revision}")
    print(f"Updated section: {args.section} in {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
