from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image

from visual_search.vlm import QwenOpenVinoVlmEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    engine = QwenOpenVinoVlmEngine(args.model, args.device, 336)
    with tempfile.TemporaryDirectory() as folder:
        image = Path(folder) / "blue.png"
        Image.new("RGB", (448, 336), (20, 80, 200)).save(image)
        verdict = engine.review("找藍色圖片", "藍色圖片", image)
        print({"match": verdict.match, "hard_fail": verdict.hard_fail, "runtime": engine.backend})


if __name__ == "__main__":
    main()
