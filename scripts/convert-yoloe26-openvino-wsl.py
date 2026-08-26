#!/usr/bin/env python3
"""Convert YOLOE-26 Prompt-Free S/M to native OpenVINO FP16 on Windows or WSL.

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
import urllib.request
from pathlib import Path


ULTRALYTICS_VERSION = "8.4.43"
ULTRALYTICS_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
PROFILES = {
    "s": ("yoloe-26s-seg-pf.pt", "yoloe-26s-pf-openvino-fp16", "yoloe_s"),
    "m": ("yoloe-26m-seg-pf.pt", "yoloe-26m-pf-openvino-fp16", "yoloe_m"),
}


def ensure_environment(no_install: bool) -> None:
    project_root = Path(__file__).resolve().parents[1]
    on_windows = platform.system() == "Windows"
    on_wsl = platform.system() == "Linux" and "microsoft" in platform.release().lower()
    if not on_windows and not on_wsl:
        raise SystemExit("This converter supports Windows and WSL2 only.")
    environment = (
        project_root / "artifacts" / "conversion-env-yoloe26"
        if on_windows else Path.home() / ".cache" / "visual-search" / "yoloe26-convert-py312"
    )
    python = environment / ("Scripts/python.exe" if on_windows else "bin/python")
    if Path(sys.prefix).resolve() != environment.resolve():
        if no_install:
            raise SystemExit(f"Run with the prepared Python 3.12 environment: {python}")
        if not python.exists():
            if on_windows:
                if sys.version_info[:2] != (3, 12):
                    raise SystemExit("Windows conversion requires Python 3.12 (the Visual Search .venv is suitable).")
                environment.parent.mkdir(parents=True, exist_ok=True)
                subprocess.check_call([sys.executable, "-m", "venv", str(environment)])
            else:
                conda = shutil.which("conda") or str(Path.home() / "miniconda3" / "bin" / "conda")
                if not Path(conda).is_file():
                    raise SystemExit("Miniconda was not found in WSL. Install Miniconda, then rerun this script.")
                subprocess.check_call([
                    conda, "create", "-y", "--override-channels", "-c", "conda-forge",
                    "-p", str(environment), "python=3.12", "pip",
                ])
        arguments = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
        if on_windows:
            # os.execv() rebuilds a Windows command line and mishandles some
            # installations when the project path contains spaces. A child
            # process preserves argv quoting correctly.
            raise SystemExit(subprocess.call(arguments))
        os.execv(str(python), arguments)

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
    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise RuntimeError("LOCALAPPDATA is not available.")
        return Path(local_app_data) / "VisualSearch" / "models"
    parts = project_root.parts
    try:
        users_index = parts.index("Users")
        profile = Path(*parts[: users_index + 2])
    except (ValueError, IndexError):
        raise RuntimeError("--install-local requires the project to be under /mnt/<drive>/Users/<name>.") from None
    return profile / "AppData" / "Local" / "VisualSearch" / "models"


def compare_raw_outputs(onnx_path: Path, openvino_path: Path, smoke: Path) -> dict[str, object]:
    """Compare native tensors before backend-specific post-processing/NMS."""
    import numpy as np
    import onnxruntime as ort
    import openvino as ov
    from PIL import Image

    image = Image.open(smoke).convert("RGB").resize((640, 640))
    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = [np.asarray(item) for item in onnx_session.run(
        None, {onnx_session.get_inputs()[0].name: tensor}
    )]
    xml = next(iter(sorted(openvino_path.glob("*.xml"))), None)
    if xml is None:
        raise RuntimeError("OpenVINO export did not contain an XML model")
    compiled = ov.Core().compile_model(xml, "CPU")
    inference = compiled([tensor])
    openvino_outputs = [np.asarray(inference[port]) for port in compiled.outputs]
    onnx_by_shape = {tuple(item.shape): item for item in onnx_outputs}
    openvino_by_shape = {tuple(item.shape): item for item in openvino_outputs}
    shapes_match = set(onnx_by_shape) == set(openvino_by_shape)
    comparisons = []
    checks = []
    if shapes_match:
        for shape in sorted(onnx_by_shape):
            left = onnx_by_shape[shape].astype(np.float32)
            right = openvino_by_shape[shape].astype(np.float32)
            difference = np.abs(left - right)
            scale = max(float(np.mean(np.abs(left))), 1e-6)
            summary = {
                "shape": list(shape),
                "max_abs_delta": float(difference.max(initial=0.0)),
                "mean_abs_delta": float(difference.mean()),
                "normalized_mean_delta": float(difference.mean() / scale),
            }
            if len(shape) == 3 and shape[-2] == 300 and shape[-1] >= 6:
                left_scores = np.sort(left[..., 4].reshape(-1))[::-1]
                right_scores = np.sort(right[..., 4].reshape(-1))[::-1]
                top = min(2, len(left_scores), len(right_scores))
                top_delta = float(np.max(np.abs(left_scores[:top] - right_scores[:top]))) if top else None
                summary.update({
                    "comparison": "top-confidence-before-runtime-filter",
                    "top_confidence_delta": top_delta,
                    "onnx_candidates_at_0_20": int(np.count_nonzero(left_scores >= 0.20)),
                    "openvino_candidates_at_0_20": int(np.count_nonzero(right_scores >= 0.20)),
                })
                checks.append(top_delta is not None and top_delta <= 0.02)
            else:
                summary["comparison"] = "full-tensor"
                checks.append(summary["normalized_mean_delta"] <= 0.01)
            checks.append(bool(np.isfinite(right).all()))
            comparisons.append(summary)
    passed = bool(comparisons) and all(checks)
    return {
        "mode": "native-tensor-before-nms",
        "shapes_match": shapes_match,
        "outputs": comparisons,
        "passed": passed,
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
    import openvino as ov
    import yaml

    weight, model_id, _section = PROFILES[scale]
    upstream = output_root.parent / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    cached_weight = upstream / weight
    if not cached_weight.is_file():
        temporary_weight = cached_weight.with_suffix(cached_weight.suffix + ".part")
        urllib.request.urlretrieve(f"{ULTRALYTICS_ASSET_BASE}/{weight}", temporary_weight)
        temporary_weight.replace(cached_weight)
    with tempfile.TemporaryDirectory(prefix=f"visual-search-yoloe-{scale}-") as temporary:
        workspace = Path(temporary)
        previous = Path.cwd()
        os.chdir(workspace)
        try:
            model = YOLOE(str(cached_weight))
            # Keep NMS outside the graph so the desktop runtime can enforce
            # class-agnostic deduplication and discard segmentation masks.
            exported_onnx = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, nms=False))
            onnx_path = workspace / exported_onnx.name
            if exported_onnx.resolve() != onnx_path.resolve():
                shutil.move(str(exported_onnx), onnx_path)
            # Convert the exact validated ONNX graph to IR. Exporting ONNX and
            # OpenVINO independently through Ultralytics can produce different
            # padded rows in the end-to-end detection tensor.
            ov_path = workspace / f"{Path(weight).stem}_openvino_model"
            ov_path.mkdir(parents=True, exist_ok=True)
            converted = ov.convert_model(onnx_path)
            ov.save_model(converted, ov_path / "model.xml", compress_to_fp16=True)
            names = getattr(model, "names", {})
            (ov_path / "metadata.yaml").write_text(
                yaml.safe_dump({
                    "task": "segment", "batch": 1, "imgsz": [640, 640],
                    "names": names,
                }, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            smoke = workspace / "visual-search-smoke.png"
            image = Image.new("RGB", (960, 640), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((90, 90, 390, 520), fill="#d3483e")
            draw.ellipse((500, 170, 830, 500), fill="#4d78b8")
            image.save(smoke)
            validation = compare_raw_outputs(onnx_path, ov_path, smoke)
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
