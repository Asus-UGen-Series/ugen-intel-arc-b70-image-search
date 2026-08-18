r"""Benchmark the production OpenVINO OCR path on one image.

Usage:
    .venv\Scripts\python.exe scripts\benchmark-ocr.py "C:\path\image.png"
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from visual_search.ocr_openvino import OpenVinoPaddleOcrEngine
from visual_search.domain import Asset
from visual_search.storage import SQLiteAssetRepository


DEFAULT_MODEL = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    / "VisualSearch"
    / "models"
    / "pp-ocrv6-medium-openvino-fp16"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--batch", type=int, default=8, choices=(8, 16, 32))
    args = parser.parse_args()

    engine = OpenVinoPaddleOcrEngine(
        args.model, device=args.device, recognition_batch_size=args.batch,
    )
    started = time.perf_counter()
    lines = engine.extract(args.image)
    elapsed = time.perf_counter() - started
    width = max((point[0] for line in lines for point in line.box), default=0)
    left = [line for line in lines if sum(point[0] for point in line.box) / max(len(line.box), 1) < width / 2]
    wanted = ("PaddleOcrEngine", "minimum_confidence", "use_doc_orientation_classify", "payload.get")
    compact_lines = ["".join(character.casefold() for character in line.text if character.isalnum()) for line in lines]
    found = {
        identifier: any(
            "".join(character.casefold() for character in identifier if character.isalnum()) in line
            for line in compact_lines
        )
        for identifier in wanted
    }
    with tempfile.TemporaryDirectory(prefix="visual-search-ocr-") as temporary:
        repository = SQLiteAssetRepository(Path(temporary) / "metadata.db")
        asset_id = repository.upsert(Asset(
            None, str(args.image), args.image.stat().st_size, args.image.stat().st_mtime_ns,
            "benchmark", int(width), 0, "",
        ))
        repository.replace_ocr(asset_id, lines, engine.last_diagnostics)
        repository.set_asset_status(asset_id, "ready")
        searchable = {identifier: bool(repository.text_search(identifier)) for identifier in wanted}
        repository.close()
    print(json.dumps({
        "image": str(args.image),
        "device": engine.device,
        "execution_devices": engine.det_execution_devices,
        "batch": engine.recognition_batch_size,
        "seconds": round(elapsed, 3),
        "lines": len(lines),
        "searchable_lines": sum(line.searchable for line in lines),
        "left_lines": len(left),
        "found": found,
        "searchable_identifiers": searchable,
        "diagnostics": engine.last_diagnostics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
