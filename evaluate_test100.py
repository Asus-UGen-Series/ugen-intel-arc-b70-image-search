from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "benchmarks" / "test100" / "images"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "artifacts" / "evaluation-reports"

LANGUAGES = {
    "1": ("中英文各跑一次｜82 個查詢（推薦）", "both"),
    "2": ("中英文交錯｜41 個查詢", "mixed"),
    "3": ("全部英文｜41 個查詢", "en"),
    "4": ("全部中文｜41 個查詢", "zh"),
}

MODES = {
    "1": ("智慧完整（推薦）", "smart_complete"),
    "2": ("徹底檢查全部圖片（最慢）", "exhaustive"),
}

THRESHOLDS = {
    "1": ("較精準（85 分）", 85),
    "2": ("平衡（70 分）", 70),
    "3": ("較多結果（60 分）", 60),
}

TOP_K = {
    "1": ("Top 5", 5),
    "2": ("Top 10（推薦）", 10),
    "3": ("Top 20", 20),
}

EXPECTED_COUNTS = {"both": 82, "mixed": 41, "en": 41, "zh": 41}
EXPECTED_TYPES = {"主體", "同義詞", "場景"}


def choose(title: str, options: dict[str, tuple[str, object]], default: str) -> object:
    print(f"\n{title}")
    for key, (label, _value) in options.items():
        marker = "  <- 預設" if key == default else ""
        print(f"  {key}. {label}{marker}")
    while True:
        answer = input(f"請選擇 [{default}]：").strip() or default
        if answer in options:
            return options[answer][1]
        print("選項無效，請重新輸入。")


def validate_benchmark(image_dir: Path) -> tuple[int, int]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from visual_search.evaluation import build_queries, load_benchmark_images

    images = load_benchmark_images(image_dir)
    if len(images) != 100:
        raise ValueError(f"test100 必須剛好有 100 張圖片，目前為 {len(images)} 張")

    for language, expected in EXPECTED_COUNTS.items():
        languages = ("en", "zh") if language == "both" else (language,)
        queries = build_queries(images, "objective", languages)
        actual_types = {query.query_type for query in queries}
        if len(queries) != expected or actual_types != EXPECTED_TYPES:
            raise ValueError(
                f"客觀題庫驗證失敗：{language} 應為 {expected} 個查詢且只包含"
                f" {sorted(EXPECTED_TYPES)}；實際為 {len(queries)}、{sorted(actual_types)}"
            )
    return len(images), EXPECTED_COUNTS["both"]


def check_environment(image_dir: Path) -> list[str]:
    errors: list[str] = []
    if not PYTHON.is_file():
        errors.append(f"找不到 Python 環境：{PYTHON}")
    if not (PROJECT_ROOT / "launcher.py").is_file():
        errors.append(f"找不到 launcher.py：{PROJECT_ROOT / 'launcher.py'}")
    if not image_dir.is_dir():
        errors.append(f"找不到圖片資料夾：{image_dir}")
    if not errors:
        try:
            validate_benchmark(image_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def build_command(
    image_dir: Path,
    report_dir: Path,
    language: str,
    mode: str,
    threshold: int,
    top_k: int,
    open_report: bool,
) -> list[str]:
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "launcher.py"),
        "--evaluate-accuracy",
        str(image_dir),
        "--evaluation-suite",
        "objective",
        "--evaluation-language",
        language,
        "--evaluation-top-k",
        str(top_k),
        "--evaluation-mode",
        mode,
        "--evaluation-threshold",
        str(threshold),
        "--evaluation-output",
        str(report_dir),
    ]
    if open_report:
        command.append("--open-report")
    return command


def pause(enabled: bool) -> None:
    if not enabled:
        return
    try:
        input("\n按 Enter 關閉視窗…")
    except EOFError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visual Search test100 客觀準確度評測")
    parser.add_argument("--check", action="store_true", help="只驗證環境、100 張圖片與 82 查詢")
    parser.add_argument("--yes", action="store_true", help="未指定的選項使用推薦值並直接開始")
    parser.add_argument("--image-dir", type=Path, help="test100 圖片資料夾")
    parser.add_argument("--report-dir", type=Path, help="本次報告輸出根目錄")
    parser.add_argument("--language", choices=("both", "mixed", "en", "zh"))
    parser.add_argument("--mode", choices=("smart_complete", "exhaustive"))
    parser.add_argument("--threshold", type=int, choices=(60, 70, 85))
    parser.add_argument("--top-k", type=int, choices=(5, 10, 20))
    parser.add_argument(
        "--open-report", action=argparse.BooleanOptionalAction, default=True,
        help="完成後開啟 HTML 報告；使用 --no-open-report 可停用",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print("Visual Search test100 客觀準確度評測")
    print("=" * 64)
    print("只測主體、同義詞與場景；標準答案來自 PNG prompt metadata。")

    if args.image_dir is not None:
        image_dir = args.image_dir.expanduser().resolve()
    elif args.yes or args.check:
        image_dir = DEFAULT_IMAGE_DIR
    else:
        value = input(f"\n圖片資料夾（直接 Enter 使用預設）\n{DEFAULT_IMAGE_DIR}\n> ").strip()
        image_dir = Path(value).expanduser().resolve() if value else DEFAULT_IMAGE_DIR

    errors = check_environment(image_dir)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        pause(not args.yes and not args.check)
        return 2

    if args.check:
        print("[PASS] Python、launcher.py、100 張圖片與客觀 82 查詢皆正常。")
        return 0

    report_dir = (args.report_dir or DEFAULT_REPORT_DIR).expanduser().resolve()
    language = args.language or ("both" if args.yes else str(choose("測試語言與查詢數", LANGUAGES, "1")))
    mode = args.mode or ("smart_complete" if args.yes else str(choose("搜尋完整度", MODES, "1")))
    threshold = args.threshold or (85 if args.yes else int(choose("結果精準度門檻", THRESHOLDS, "1")))
    top_k = args.top_k or (10 if args.yes else int(choose("排名評測範圍", TOP_K, "2")))
    command = build_command(
        image_dir, report_dir, language, mode, threshold, top_k, args.open_report,
    )

    print("\n即將執行：")
    print(f"  {EXPECTED_COUNTS[language]} 個客觀查詢（主體、同義詞、場景）")
    print(f"  {mode} / 門檻 {threshold} / Top {top_k}")
    print(f"  圖片：{image_dir}")
    print(f"  報告：{report_dir}")
    print("\n請先關閉 Visual Search 主程式，避免兩個程序同時占用 GPU。")
    if not args.yes:
        answer = input("按 Enter 開始；輸入 q 取消：").strip().casefold()
        if answer == "q":
            print("已取消。")
            return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode == 0:
        print("\n[PASS] 客觀評測完成。")
    else:
        print(f"\n[FAIL] 評測未完整完成，錯誤碼：{completed.returncode}")
    pause(not args.yes)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
