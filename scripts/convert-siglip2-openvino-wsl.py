#!/usr/bin/env python3
"""Convert SigLIP 2 SO400M NaFlex to OpenVINO FP16 directly in WSL.

The converter creates a dedicated Python 3.12 environment under ~/.cache,
installs conversion-only dependencies there, delegates the validated export to
export-siglip.py, and can copy the result into Windows LOCALAPPDATA. It never
modifies the project's Windows .venv.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


MODEL_ID = "siglip2-so400m-patch16-naflex-openvino-fp16"
PYTHON_DEPENDENCIES = (
    "openvino>=2026.0,<2027.0",
    "transformers>=4.56,<5",
    "accelerate>=1.2,<2",
    "huggingface-hub>=0.34,<2",
    "numpy>=1.26,<3",
    "pillow>=10.4",
    "sentencepiece>=0.2",
    "protobuf>=5",
    "safetensors>=0.5",
)


def ensure_wsl_environment(no_install: bool) -> None:
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise SystemExit("This converter must be run from a WSL terminal.")

    environment = Path.home() / ".cache" / "visual-search" / "siglip2-convert-py312"
    environment_python = environment / "bin" / "python"
    running_inside_environment = Path(sys.prefix).resolve() == environment.resolve()
    supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    if not supported or not running_inside_environment:
        if no_install:
            raise SystemExit("Use Python 3.11-3.13 with the required conversion dependencies.")
        conda = shutil.which("conda") or str(Path.home() / "miniconda3" / "bin" / "conda")
        if not Path(conda).is_file():
            raise SystemExit("Miniconda was not found in WSL. Install Miniconda or provide Python 3.12.")
        if not environment_python.exists():
            print(f"Creating isolated Python 3.12 environment: {environment}")
            subprocess.check_call([
                conda, "create", "-y", "--override-channels", "-c", "conda-forge",
                "-p", str(environment), "python=3.12", "pip",
            ])
        os.execv(str(environment_python), [str(environment_python), str(Path(__file__).resolve()), *sys.argv[1:]])

    required_modules = ("torch", "openvino", "transformers", "accelerate", "sentencepiece")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if no_install:
        raise SystemExit(f"Missing conversion dependencies: {', '.join(missing)}")
    print("Installing CPU-only PyTorch for model conversion...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--index-url",
        "https://download.pytorch.org/whl/cpu", "torch>=2.6,<3",
    ])
    print("Installing SigLIP 2 and OpenVINO conversion dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *PYTHON_DEPENDENCIES])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows_profile_from_project(project_root: Path) -> Path:
    parts = project_root.parts
    try:
        users_index = parts.index("Users")
        return Path(*parts[: users_index + 2])
    except (ValueError, IndexError):
        raise RuntimeError(
            "Project must be under /mnt/<drive>/Users/<name> to use --install-local."
        ) from None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project_root / "release-models" / MODEL_ID)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-local", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()
    ensure_wsl_environment(args.no_install)

    output = args.output.resolve()
    command = [
        sys.executable,
        str(project_root / "scripts" / "export-siglip.py"),
        "--output", str(output),
    ]
    if args.force:
        command.append("--force")
    subprocess.check_call(command, cwd=project_root)

    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and "compiled_cache" not in item.parts):
        files.append({
            "name": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    package = {"model_id": MODEL_ID, "files": files}
    (output / "package-manifest.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.install_local:
        destination = windows_profile_from_project(project_root) / "AppData" / "Local" / "VisualSearch" / "models" / MODEL_ID
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(output, destination)
        print(f"Installed local development package: {destination}")

    print(f"Conversion complete: {output}")
    print("Next: python scripts/publish-model.py --section model --repo <account>/<repo>")


if __name__ == "__main__":
    main()
