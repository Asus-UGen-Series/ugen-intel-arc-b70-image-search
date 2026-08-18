# -*- mode: python ; coding: utf-8 -*-
import hashlib
from pathlib import PureWindowsPath

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []
for package in ("openvino", "openvino_genai", "openvino_tokenizers", "usearch"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

# PyInstaller's maintained hooks collect Torch/Torchvision native libraries.
# Only include the dynamic Transformers and Optimum modules used by Qwen VLM,
# instead of sweeping every architecture, test, and conversion utility.
for package in (
    "optimum.intel.openvino",
    "transformers.models.auto",
    "transformers.models.qwen3_5",
    "transformers.models.qwen3_vl",
    "transformers.generation",
):
    hiddenimports += collect_submodules(package)

for distribution in (
    "openvino", "openvino-genai", "openvino-tokenizers", "optimum",
    "optimum-intel", "transformers",
    "huggingface-hub", "usearch",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

a = Analysis(
    ["launcher.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas + [("src/visual_search/resources", "visual_search/resources")],
    hiddenimports=hiddenimports + ["visual_search.ui"],
    excludes=["tensorflow", "paddle", "paddleocr", "paddlex"],
)

# PyInstaller's Torch hook preserves a very deep third-party license tree.
# Flatten those license destinations so installation also works for Windows
# accounts and custom paths that would otherwise exceed MAX_PATH. Contents are
# retained; only their destination names change.
flattened_datas = []
for destination, source, typecode in a.datas:
    normalized = destination.replace("/", "\\")
    if normalized.startswith("torch-") and "\\licenses\\third_party\\" in normalized:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        name = PureWindowsPath(normalized).name[:64]
        destination = f"licenses\\torch-third-party\\{digest}-{name}"
    flattened_datas.append((destination, source, typecode))
a.datas = flattened_datas

pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VisualSearch", console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="VisualSearch")
