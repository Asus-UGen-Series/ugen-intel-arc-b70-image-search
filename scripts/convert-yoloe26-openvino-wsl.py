#!/usr/bin/env python3
"""Convert YOLOE-26 Prompt-Free S/M to native OpenVINO FP16 in WSL.

Ultralytics and ONNX Runtime are conversion-only dependencies. They are never
imported by Visual Search at runtime and the generated ONNX file is removed
after the OpenVINO comparison unless --keep-onnx is supplied.
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
import tempfile
from pathlib import Path


ULTRALYTICS_VERSION = "8.4.43"
PROFILES = {
    "s": ("yoloe-26s-seg-pf.pt", "yoloe-26s-pf-openvino-fp16", "yoloe_s"),
    "m": ("yoloe-26m-seg-pf.pt", "yoloe-26m-pf-openvino-fp16", "yoloe_m"),
}


def ensure_environment(no_install: bool) -> None:
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise SystemExit("This converter must be run inside WSL2.")
    environment = Path.home() / ".cache" / "visual-search" / "yoloe26-convert-py312"
    python = environment / "bin" / "python"
    if Path(sys.prefix).resolve() != environment.resolve():
        if no_install:
            raise SystemExit(f"Run with the prepared Python 3.12 environment: {python}")
        conda = shutil.which("conda") or str(Path.home() / "miniconda3" / "bin" / "conda")
        if not Path(conda).is_file():
            raise SystemExit("Miniconda was not found in WSL. Install Miniconda, then rerun this script.")
        if not python.exists():
            subprocess.check_call([
                conda, "create", "-y", "--override-channels", "-c", "conda-forge",
                "-p", str(environment), "python=3.12", "pip",
            ])
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])

    required = ("torch", "ultralytics", "openvino", "onnxruntime", "yaml")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        import ultralytics
        if ultralytics.__version__ == ULTRALYTICS_VERSION:
            return
    if no_install:
        raise SystemExit("Conversion dependencies are missing or have the wrong Ultralytics version.")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--index-url",
        "https://download.pytorch.org/whl/cpu", "torch>=2.6,<3", "torchvision>=0.21,<1",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        f"ultralytics=={ULTRALYTICS_VERSION}", "openvino>=2026.0,<2027.0",
        "onnxruntime>=1.20,<2", "onnx>=1.17,<2", "pillow>=10.4", "pyyaml>=6,<7",
    ])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_replace_tree(source: Path, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"Output exists; pass --force to replace it: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.onnx", "compiled_cache"))


def windows_local_models(project_root: Path) -> Path:
    parts = project_root.parts
    try:
        users_index = parts.index("Users")
        profile = Path(*parts[: users_index + 2])
    except (ValueError, IndexError):
        raise RuntimeError("--install-local requires the project to be under /mnt/<drive>/Users/<name>.") from None
    return profile / "AppData" / "Local" / "VisualSearch" / "models"


def comparison_summary(onnx_results, openvino_results) -> dict[str, object]:
    onnx_boxes = onnx_results[0].boxes
    ov_boxes = openvino_results[0].boxes
    onnx_count = 0 if onnx_boxes is None else len(onnx_boxes)
    ov_count = 0 if ov_boxes is None else len(ov_boxes)
    score_delta = None
    if onnx_count and ov_count:
        left = onnx_boxes.conf.detach().cpu().numpy()
        right = ov_boxes.conf.detach().cpu().numpy()
        count = min(len(left), len(right))
        score_delta = float(abs(left[:count] - right[:count]).max())
    return {
        "onnx_detections": onnx_count,
        "openvino_detections": ov_count,
        "max_score_delta": score_delta,
        "passed": abs(onnx_count - ov_count) <= 2 and (score_delta is None or score_delta <= 0.08),
    }


def write_package_files(destination: Path, scale: str, weight: str, validation: dict[str, object]) -> None:
    import yaml

    metadata_path = destination / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    names = metadata.get("names", {})
    if isinstance(names, dict):
        ordered = [str(value) for _key, value in sorted(names.items(), key=lambda item: int(item[0]))]
    else:
        ordered = [str(value) for value in names]
    metadata.update({
        "visual_search_family": "yoloe26_pf",
        "visual_search_profile": f"yoloe26{scale}-pf",
        "prompt_mode": "prompt-free",
        "ultralytics_version": ULTRALYTICS_VERSION,
        "source_weight": weight,
        "scale": scale.upper(),
    })
    metadata_path.write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (destination / "vocabulary.json").write_text(
        json.dumps({"labels": ordered}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "---\nlicense: agpl-3.0\nlibrary_name: openvino\ntags:\n- openvino\n- yoloe\n- prompt-free\n---\n\n"
        f"# YOLOE-26{scale.upper()} Prompt-Free OpenVINO FP16\n\n"
        f"Converted from `{weight}` with Ultralytics `{ULTRALYTICS_VERSION}` for Visual Search. "
        "The exported vocabulary is fixed at conversion time. Runtime inference uses native OpenVINO; "
        "Ultralytics and ONNX Runtime are not required by the desktop application.\n\n"
        "License: AGPL-3.0. Review the upstream Ultralytics terms before redistribution.\n",
        encoding="utf-8",
    )
    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file() and item.name != "package-manifest.json"):
        files.append({
            "name": path.relative_to(destination).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    (destination / "package-manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "model_id": destination.name,
            "runtime": "OpenVINO FP16",
            "prompt_mode": "prompt-free",
            "files": files,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert(scale: str, output_root: Path, keep_onnx: bool, force: bool) -> Path:
    from PIL import Image, ImageDraw
    from ultralytics import YOLOE

    weight, model_id, _section = PROFILES[scale]
    with tempfile.TemporaryDirectory(prefix=f"visual-search-yoloe-{scale}-") as temporary:
        workspace = Path(temporary)
        previous = Path.cwd()
        os.chdir(workspace)
        try:
            model = YOLOE(weight)
            # Keep NMS outside the graph so the desktop runtime can enforce
            # class-agnostic deduplication and discard segmentation masks.
            onnx_path = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, nms=False))
            ov_path = Path(model.export(format="openvino", imgsz=640, dynamic=False, quantize=16, nms=False))
            smoke = workspace / "visual-search-smoke.png"
            image = Image.new("RGB", (960, 640), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((90, 90, 390, 520), fill="#d3483e")
            draw.ellipse((500, 170, 830, 500), fill="#4d78b8")
            image.save(smoke)
            onnx_results = YOLOE(str(onnx_path)).predict(source=str(smoke), imgsz=640, verbose=False)
            ov_results = YOLOE(str(ov_path)).predict(source=str(smoke), imgsz=640, verbose=False)
            validation = comparison_summary(onnx_results, ov_results)
            if not validation["passed"]:
                raise RuntimeError(f"ONNX/OpenVINO comparison failed: {validation}")
            destination = output_root / model_id
            safe_replace_tree(ov_path, destination, force)
            write_package_files(destination, scale, weight, validation)
            if keep_onnx:
                comparison = output_root.parent / "conversion-only" / model_id
                comparison.mkdir(parents=True, exist_ok=True)
                shutil.copy2(onnx_path, comparison / onnx_path.name)
                shutil.copy2(smoke, comparison / smoke.name)
            return destination
        finally:
            os.chdir(previous)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("s", "m", "both"), default="both")
    parser.add_argument("--output", type=Path, default=project_root / "artifacts" / "hf")
    parser.add_argument("--keep-onnx", action="store_true")
    parser.add_argument("--install-local", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()
    ensure_environment(args.no_install)
    scales = ("s", "m") if args.scale == "both" else (args.scale,)
    for scale in scales:
        destination = convert(scale, args.output.resolve(), args.keep_onnx, args.force)
        section = PROFILES[scale][2]
        if args.install_local:
            local = windows_local_models(project_root) / destination.name
            if local.exists():
                shutil.rmtree(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(destination, local)
            print(f"Installed local development package: {local}")
        print(f"Converted {scale.upper()}: {destination}")
        print(f"Prepare upload: python scripts/prepare-hf-upload.py --folder '{destination}' --repo owner/repo")
        print(f"Pin after upload: python scripts/finalize-model-source.py --section {section} --repo owner/repo")


if __name__ == "__main__":
    main()
