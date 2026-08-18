#!/usr/bin/env python3
"""Download official PP-OCRv6 ONNX models and convert them to OpenVINO IR.

Designed to run directly inside WSL from the repository mounted under /mnt/c.
It installs conversion-only Python dependencies into the active environment on
first use; it never installs PaddlePaddle and never modifies the Windows venv.
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


DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
DET_REVISION = "61323801669c338b7891481ec7bac61ce31b576a"
REC_REPO = "PaddlePaddle/PP-OCRv6_medium_rec_onnx"
REC_REVISION = "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"
MODEL_ID = "pp-ocrv6-medium-openvino-fp16"
DEPENDENCIES = (
    "openvino>=2026.0,<2027.0",
    "huggingface-hub>=0.34,<2",
    "numpy>=1.26,<3",
    "pyyaml>=6.0",
)


def ensure_environment(no_install: bool) -> None:
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise SystemExit("This converter is intended for WSL. Run it from an Ubuntu/WSL terminal.")
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        if no_install:
            raise SystemExit(
                f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported. "
                "Use WSL Python 3.11, 3.12, or 3.13."
            )
        conda = shutil.which("conda") or str(Path.home() / "miniconda3" / "bin" / "conda")
        if not Path(conda).is_file():
            raise SystemExit(
                f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported and Miniconda was not found. "
                "Install Python 3.12 in WSL, then run this file again."
            )
        environment = Path.home() / ".cache" / "visual-search" / "openvino-convert-py312"
        python = environment / "bin" / "python"
        if not python.exists():
            print(f"Creating isolated WSL Python 3.12 environment at {environment}…")
            subprocess.check_call([
                conda, "create", "-y", "--override-channels", "-c", "conda-forge",
                "-p", str(environment), "python=3.12", "pip",
            ])
        print("Restarting converter with WSL Python 3.12…")
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    missing = [name for name in ("openvino", "huggingface_hub", "yaml", "numpy") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if no_install:
        raise SystemExit(f"Missing conversion dependencies: {', '.join(missing)}")
    print("Installing WSL conversion dependencies into the active environment…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *DEPENDENCIES])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(repo: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"Downloading {repo}@{revision[:8]}/{filename}…")
    return Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))


def convert(onnx_path: Path, destination: Path, input_name: str) -> None:
    import openvino as ov

    model = ov.convert_model(onnx_path)
    model.input(0).get_tensor().set_names({input_name})
    model.output(0).get_tensor().set_names({"probabilities"})
    ov.save_model(model, destination, compress_to_fp16=True)


def smoke_test(output: Path, device: str) -> dict[str, object]:
    import numpy as np
    import openvino as ov

    core = ov.Core()
    available = core.available_devices
    if device != "CPU" and device not in available:
        raise RuntimeError(f"Requested verification device {device} is unavailable: {available}")
    report: dict[str, object] = {"requested_device": device, "available_devices": available, "models": {}}
    cases = {
        "det": (output / "det_model.xml", np.zeros((1, 3, 960, 960), dtype=np.float32)),
        "rec": (output / "rec_model.xml", np.zeros((1, 3, 48, 320), dtype=np.float32)),
    }
    for name, (path, sample) in cases.items():
        compiled = core.compile_model(path, device, {"PERFORMANCE_HINT": "LATENCY"})
        result = np.asarray(compiled({compiled.input(0): sample})[compiled.output(0)])
        if not result.size or not np.isfinite(result).all():
            raise RuntimeError(f"{name} smoke test returned invalid data")
        try:
            execution_devices = [str(item) for item in compiled.get_property("EXECUTION_DEVICES")]
        except Exception:
            execution_devices = [device]
        report["models"][name] = {
            "input_shape": list(sample.shape),
            "output_shape": list(result.shape),
            "execution_devices": execution_devices,
        }
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project_root / "release-models" / MODEL_ID)
    parser.add_argument("--verify-device", default="CPU", help="CPU by default; use GPU only if OpenVINO GPU is visible in WSL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--install-local",
        action="store_true",
        help="also copy the generated package into the current Windows user's LOCALAPPDATA VisualSearch model folder",
    )
    parser.add_argument("--no-install", action="store_true", help="fail instead of installing missing Python packages")
    args = parser.parse_args()
    ensure_environment(args.no_install)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise SystemExit(f"Output is not empty: {output}\nRe-run with --force to replace generated files.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    det_onnx = download(DET_REPO, DET_REVISION, "inference.onnx")
    det_yml = download(DET_REPO, DET_REVISION, "inference.yml")
    rec_onnx = download(REC_REPO, REC_REVISION, "inference.onnx")
    rec_yml = download(REC_REPO, REC_REVISION, "inference.yml")

    print("Converting PP-OCRv6 detection to OpenVINO FP16…")
    convert(det_onnx, output / "det_model.xml", "x")
    print("Converting PP-OCRv6 recognition to OpenVINO FP16…")
    convert(rec_onnx, output / "rec_model.xml", "x")
    shutil.copyfile(det_yml, output / "det_inference.yml")
    shutil.copyfile(rec_yml, output / "rec_inference.yml")

    print(f"Running OpenVINO smoke tests on {args.verify_device}…")
    verification = smoke_test(output, args.verify_device)
    metadata = {
        "format": "visual-search-openvino-ocr-v1",
        "model_id": MODEL_ID,
        "precision": "FP16",
        "upstream": {
            "detection": {"repo": DET_REPO, "revision": DET_REVISION},
            "recognition": {"repo": REC_REPO, "revision": REC_REVISION},
        },
        "verification": verification,
    }
    (output / "openvino_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    files = []
    for path in sorted(item for item in output.iterdir() if item.is_file() and item.name != "package-manifest.json"):
        files.append({"name": path.name, "size": path.stat().st_size, "sha256": digest(path)})
    package = {"model_id": MODEL_ID, "files": files, "metadata": metadata}
    (output / "package-manifest.json").write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    shutil.rmtree(output / "compiled_cache", ignore_errors=True)
    if args.install_local:
        parts = project_root.parts
        try:
            users_index = parts.index("Users")
            profile = Path(*parts[: users_index + 2])
        except (ValueError, IndexError):
            raise RuntimeError("Project is not under /mnt/<drive>/Users/<name>; copy the output folder manually.")
        destination = profile / "AppData" / "Local" / "VisualSearch" / "models" / MODEL_ID
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(output, destination)
        print(f"Installed local development package: {destination}")
    print(f"\nConversion complete: {output}")
    print("Next: upload this folder with scripts/publish-model.py --section ocr_model")


if __name__ == "__main__":
    main()
