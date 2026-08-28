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
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--query", default="找藍色圖片")
    args = parser.parse_args()
    engine = QwenOpenVinoVlmEngine(args.model, args.device, 336)
    with tempfile.TemporaryDirectory() as folder:
        images = [path for path in args.image[:args.batch] if path.is_file()]
        if not images:
            colors = ((20, 80, 200), (200, 30, 30), (20, 160, 70), (120, 120, 120))
            for index, color in enumerate(colors[:args.batch]):
                image = Path(folder) / f"color-{index}.png"
                Image.new("RGB", (448, 336), color).save(image)
                images.append(image)
        verdicts = engine.review_batch(
            args.query, args.query, images, [() for _image in images],
        )
        print({
            "batch": args.batch,
            "verdicts": [
                {"candidate_id": row.candidate_id, "match": row.verdict.match,
                 "hard_fail": row.verdict.hard_fail}
                for row in verdicts
            ],
            "runtime": engine.backend,
            "device": engine.device,
        })


if __name__ == "__main__":
    main()
