"""Validate an already-uploaded Hugging Face repo and pin it in models.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, hf_hub_url


REQUIRED = {
    "vlm_int8": {"config.json", "processor_config.json", "openvino_language_model.xml", "openvino_language_model.bin"},
    "vlm_int4": {"config.json", "processor_config.json", "openvino_language_model.xml", "openvino_language_model.bin"},
    "yoloe_s": {"metadata.yaml"},
    "yoloe_m": {"metadata.yaml"},
    "model": {"vision_model.xml", "vision_model.bin", "text_model.xml", "text_model.bin", "position_embedding.npy"},
    "ocr_model": {"det_model.xml", "det_model.bin", "rec_model.xml", "rec_model.bin", "rec_inference.yml"},
}

REPO_PATHS = {
    "model": "siglip2-so400m-patch16-naflex-openvino-fp16",
    "ocr_model": "pp-ocrv6-medium-openvino-fp16",
    "yoloe_s": "yoloe-26s-pf-openvino-fp16",
    "yoloe_m": "yoloe-26m-pf-openvino-fp16",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--section", choices=REQUIRED, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("src/visual_search/resources/models.json"))
    args = parser.parse_args()
    api = HfApi()
    info = api.model_info(args.repo, revision="main", files_metadata=True)
    names = {item.rfilename for item in info.siblings}
    prefix = ""
    if not REQUIRED[args.section] <= names and args.section in REPO_PATHS:
        candidate = REPO_PATHS[args.section] + "/"
        prefixed_names = {
            name[len(candidate):] for name in names if name.startswith(candidate)
        }
        if REQUIRED[args.section] <= prefixed_names:
            prefix = candidate
    local_names = {name[len(prefix):] for name in names if not prefix or name.startswith(prefix)}
    missing = REQUIRED[args.section] - local_names
    if args.section.startswith("yoloe_"):
        if not any(name.endswith(".xml") for name in local_names):
            missing.add("<OpenVINO .xml>")
        if not any(name.endswith(".bin") for name in local_names):
            missing.add("<OpenVINO .bin>")
    if missing:
        raise SystemExit("遠端模型不完整：" + ", ".join(sorted(missing)))
    files = []
    for item in info.siblings:
        if prefix and not item.rfilename.startswith(prefix):
            continue
        name = item.rfilename[len(prefix):] if prefix else item.rfilename
        if name in {".gitattributes", "README.md"}:
            continue
        lfs = getattr(item, "lfs", None)
        digest = getattr(lfs, "sha256", None)
        if not digest:
            response = requests.get(hf_hub_url(args.repo, item.rfilename, revision=info.sha), timeout=90)
            response.raise_for_status()
            digest = hashlib.sha256(response.content).hexdigest()
        files.append({
            "name": name,
            "source_name": item.rfilename,
            "url": f"https://huggingface.co/{args.repo}/resolve/{info.sha}/{quote(item.rfilename)}?download=true",
            "sha256": digest, "size": item.size,
        })
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry = dict(manifest.get(args.section, {}))
    entry.update({"source_repo": args.repo, "revision": info.sha, "files": files})
    manifest["manifest_version"] = 5
    manifest[args.section] = entry
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned {args.repo}@{info.sha} in {args.section}")


if __name__ == "__main__":
    main()
