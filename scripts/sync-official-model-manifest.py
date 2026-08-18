"""Pin official OpenVINO factory models and write per-file SHA-256 metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, hf_hub_url


FACTORY_MODELS = {
    "vlm_int8": {
        "id": "qwen3.5-9b-int8-openvino", "directory": "qwen3.5-9b-int8-openvino",
        "repo": "OpenVINO/Qwen3.5-9B-int8-ov", "upstream": "Qwen/Qwen3.5-9B",
        "precision": "INT8", "family": "qwen3_5",
        "required": ["openvino_language_model.xml", "openvino_language_model.bin", "config.json", "processor_config.json"],
    },
    "vlm_int4": {
        "id": "qwen3.5-9b-int4-openvino", "directory": "qwen3.5-9b-int4-openvino",
        "repo": "OpenVINO/Qwen3.5-9B-int4-ov", "upstream": "Qwen/Qwen3.5-9B",
        "precision": "INT4", "family": "qwen3_5",
        "required": ["openvino_language_model.xml", "openvino_language_model.bin", "config.json", "processor_config.json"],
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def pinned_entry(api: HfApi, specification: dict[str, object]) -> dict[str, object]:
    repo = str(specification["repo"])
    info = api.model_info(repo, files_metadata=True)
    result: list[dict[str, object]] = []
    for sibling in info.siblings:
        if sibling.rfilename in {".gitattributes", "README.md"}:
            continue
        lfs = getattr(sibling, "lfs", None)
        digest = getattr(lfs, "sha256", None)
        if not digest:
            response = requests.get(hf_hub_url(repo, sibling.rfilename, revision=info.sha), timeout=90)
            response.raise_for_status()
            if len(response.content) != sibling.size:
                raise RuntimeError(f"Size changed while pinning {repo}/{sibling.rfilename}")
            digest = sha256_bytes(response.content)
        name = sibling.rfilename
        result.append({
            "name": name,
            "url": f"https://huggingface.co/{repo}/resolve/{info.sha}/{quote(name)}?download=true",
            "sha256": digest,
            "size": sibling.size,
        })
    entry = {key: value for key, value in specification.items() if key != "repo"}
    entry.update({"source_repo": repo, "revision": info.sha, "files": result})
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("src/visual_search/resources/models.json"))
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}
    payload["manifest_version"] = 5
    api = HfApi()
    for section, specification in FACTORY_MODELS.items():
        print(f"Pinning {specification['repo']}…", flush=True)
        payload[section] = pinned_entry(api, specification)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.manifest.resolve())


if __name__ == "__main__":
    main()
