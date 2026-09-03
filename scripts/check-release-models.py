"""Validate that every production model has an immutable, checksummed source."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "src" / "visual_search" / "resources" / "models.json"
REQUIRED_SECTIONS = ("model", "ocr_model", "yoloe_s", "yoloe_m", "vlm_int8", "vlm_int4")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def safe_relative(value: object) -> bool:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    return bool(text) and not path.is_absolute() and ".." not in path.parts


def validate_entry(section: str, entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return [f"{section}: 缺少 manifest section"]
    errors: list[str] = []
    repo = str(entry.get("source_repo") or "")
    revision = str(entry.get("revision") or "")
    files = entry.get("files")
    if not repo or "/" not in repo:
        errors.append(f"{section}: source_repo 尚未設定")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append(f"{section}: revision 必須是 40 字元 immutable commit")
    if not isinstance(files, list) or not files:
        errors.append(f"{section}: files 尚未設定")
        return errors

    names: set[str] = set()
    for index, item in enumerate(files, 1):
        label = f"{section}.files[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label}: 必須是物件")
            continue
        name = item.get("name")
        source_name = item.get("source_name") or name
        if not safe_relative(name):
            errors.append(f"{label}: name 不是安全相對路徑")
        elif str(name) in names:
            errors.append(f"{label}: name 重複：{name}")
        else:
            names.add(str(name))
        if not safe_relative(source_name):
            errors.append(f"{label}: source_name 不是安全相對路徑")
        if not SHA256_PATTERN.fullmatch(str(item.get("sha256") or "")):
            errors.append(f"{label}: sha256 無效")
        if not isinstance(item.get("size"), int) or int(item.get("size") or 0) <= 0:
            errors.append(f"{label}: size 必須大於 0")
        expected_prefix = f"https://huggingface.co/{repo}/resolve/{revision}/"
        if repo and revision and not str(item.get("url") or "").startswith(expected_prefix):
            errors.append(f"{label}: URL 未固定至指定 repo/revision")

    required = {str(name) for name in entry.get("required", [])}
    missing = required - names
    if missing:
        errors.append(f"{section}: files 缺少必要檔案：{', '.join(sorted(missing))}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="檢查正式版模型下載 manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="允許尚未發布的自製模型缺漏，但仍檢查已設定項目",
    )
    args = parser.parse_args(argv)
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_errors: list[str] = []
    incomplete: list[str] = []
    for section in REQUIRED_SECTIONS:
        errors = validate_entry(section, data.get(section))
        if errors and args.allow_incomplete and section in {"model", "ocr_model", "yoloe_s", "yoloe_m"}:
            entry = data.get(section) or {}
            if not entry.get("source_repo") and not entry.get("revision") and not entry.get("files"):
                incomplete.append(section)
                continue
        all_errors.extend(errors)

    if all_errors:
        for error in all_errors:
            print(f"[FAIL] {error}")
        return 1
    if incomplete:
        print("[WARN] 正式發布仍缺少自製模型：" + ", ".join(incomplete))
        print("       建立 Hugging Face 模型庫後執行 scripts/publish-model.py --bundle。")
    else:
        print("[PASS] 所有模型皆有 immutable revision、大小與 SHA-256。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
