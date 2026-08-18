#!/usr/bin/env python3
"""Convert Qwen3.5-9B to a publishable OpenVINO INT8 or INT4 package in WSL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path


UPSTREAM = "Qwen/Qwen3.5-9B"
UPSTREAM_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_manifest(folder: Path, precision: str) -> None:
    files = []
    for path in sorted(item for item in folder.rglob("*") if item.is_file() and "compiled_cache" not in item.parts):
        if path.name == "package-manifest.json":
            continue
        files.append({
            "name": path.relative_to(folder).as_posix(), "size": path.stat().st_size, "sha256": sha256(path),
        })
    payload = {
        "format": "OpenVINO IR", "upstream": UPSTREAM,
        "upstream_revision": UPSTREAM_REVISION, "precision": precision,
        "quantization": "INT8_ASYM group_size=-1" if precision == "INT8" else "INT4_ASYM group_size=128 ratio=1.0",
        "files": files,
    }
    (folder / "package-manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def windows_install_path(project: Path, directory: str) -> Path:
    parts = project.resolve().parts
    try:
        users = parts.index("Users")
        username = parts[users + 1]
    except (ValueError, IndexError):
        raise RuntimeError("無法從專案路徑判斷 Windows 使用者；請改用 --install-dir")
    return Path("/mnt/c/Users") / username / "AppData/Local/VisualSearch/models" / directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-format", choices=("int8", "int4"), default="int8")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/hf"))
    parser.add_argument("--venv", type=Path, default=Path(".convert-venvs/qwen35-openvino"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-local", action="store_true")
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or "microsoft" not in platform.release().casefold():
        raise SystemExit("此工具設計為在 WSL2 執行。")
    project = Path(__file__).resolve().parents[1]
    precision = args.weight_format.upper()
    directory = f"Qwen3.5-9B-{args.weight_format}-ov"
    output = (project / args.output_root / directory).resolve()
    environment = (project / args.venv).resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"輸出已存在：{output}；使用 --force 才會重建。")
        shutil.rmtree(output)
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    run([str(python), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    run([
        str(python), "-m", "pip", "install", "-U",
        "torch>=2.7,<3", "torchvision>=0.22,<1", "transformers==5.2.*", "accelerate>=1.2",
        "nncf>=2.18", "openvino>=2026.2,<2027", "openvino-genai>=2026.2,<2027",
        "https://github.com/huggingface/optimum-intel/archive/c1efc7dc3e6650ae7bf68fdab7488f2261400c98.zip",
        "Pillow", "numpy",
    ])
    checkpoint = project / ".model-cache" / f"Qwen3.5-9B-{UPSTREAM_REVISION[:12]}"
    if not (checkpoint / "config.json").is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        hf = environment / "bin/hf"
        run([
            str(hf), "download", UPSTREAM, "--revision", UPSTREAM_REVISION,
            "--local-dir", str(checkpoint),
        ])
    optimum = environment / "bin/optimum-cli"
    command = [
        str(optimum), "export", "openvino", "--model", str(checkpoint),
        "--task", "image-text-to-text", "--weight-format", args.weight_format,
    ]
    if args.weight_format == "int8":
        command += ["--group-size", "-1"]
    else:
        command += ["--group-size", "128", "--ratio", "1.0"]
    command.append(str(output))
    run(command)
    required = [
        "config.json", "processor_config.json", "tokenizer_config.json",
        "openvino_language_model.xml", "openvino_language_model.bin",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing or not list(output.glob("openvino_vision*_model.xml")):
        raise RuntimeError("轉換結果不完整：" + ", ".join(missing or ["vision model"]))
    package_manifest(output, precision)
    (output / "README.md").write_text(
        f"# Qwen3.5-9B {precision} OpenVINO\n\n"
        f"Converted from `{UPSTREAM}` at `{UPSTREAM_REVISION}` for Visual Search.\n\n"
        "Original model license: Apache-2.0. This repository contains OpenVINO IR, "
        "tokenizer and processor files and does not require remote Python code at runtime.\n",
        encoding="utf-8",
    )
    if not args.skip_smoke:
        smoke = project / "scripts" / "smoke-vlm-openvino.py"
        run([str(python), str(smoke), "--model", str(output), "--device", "CPU"],
            env={**os.environ, "PYTHONPATH": str(project / "src")})
    if args.install_local:
        target = args.install_dir or windows_install_path(project, f"qwen3.5-9b-{args.weight_format}-openvino")
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(output, target, ignore=shutil.ignore_patterns("compiled_cache"))
        print(f"Windows 開發版模型：{target}")
    print(f"OpenVINO 輸出：{output}")


if __name__ == "__main__":
    main()
