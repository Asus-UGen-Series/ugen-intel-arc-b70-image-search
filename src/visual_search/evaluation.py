from __future__ import annotations

import csv
import html
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image

from .domain import SearchHit, SearchMode


IMAGE_ID_PATTERN = re.compile(r"^(\d{8})_")

OBJECT_ZH = {
    "a vintage camera": "老式相機",
    "a golden retriever": "黃金獵犬",
    "a chocolate birthday cake": "巧克力生日蛋糕",
    "a wooden dining chair": "木製餐椅",
    "intricate circuit board": "精密電路板",
    "a yellow school bus": "黃色校車",
    "a passenger airplane": "客機",
    "a red sports car": "紅色跑車",
    "a bowl of ramen": "一碗拉麵",
    "a red umbrella": "紅色雨傘",
    "a green hiking backpack": "綠色登山背包",
    "a glass perfume bottle": "玻璃香水瓶",
    "a violin": "小提琴",
    "a ceramic coffee mug": "陶瓷咖啡杯",
    "a cargo ship": "貨輪",
    "digital calipers": "數位卡尺",
    "an oscilloscope": "示波器",
    "a tropical houseplant": "熱帶室內植物",
    "a black cat": "黑貓",
    "a desktop computer": "桌上型電腦",
    "a blue city bicycle": "藍色城市腳踏車",
}

SCENE_ZH = {
    "at an international airport": "國際機場",
    "in a dense green forest": "茂密綠色森林",
    "at an outdoor market": "戶外市場",
    "in a snowy mountain village": "雪山村落",
    "on a marble museum pedestal": "大理石博物館展示台",
    "on a busy Tokyo street": "繁忙東京街頭",
    "in laboratory": "實驗室",
    "in a clean photo studio": "乾淨攝影棚",
    "inside a warehouse": "倉庫內",
    "on a quiet beach": "安靜海灘",
    "inside a modern apartment": "現代公寓內",
}

ALIASES = (
    ("car", "汽車", "a red sports car"),
    ("dog", "狗", "a golden retriever"),
    ("cat", "貓", "a black cat"),
    ("oscilloscope", "量測波形的儀器", "an oscilloscope"),
    ("circuit board", "電路板", "intricate circuit board"),
    ("perfume", "香水", "a glass perfume bottle"),
    ("cup", "杯子", "a ceramic coffee mug"),
    ("cargo ship", "船", "a cargo ship"),
    ("airplane", "飛機", "a passenger airplane"),
)

OBJECTIVE_QUERY_TYPES = frozenset({"主體", "同義詞", "場景"})
SUPPORTED_SUITES = frozenset({"objective", "core", "full"})


@dataclass(frozen=True, slots=True)
class BenchmarkImage:
    image_id: int
    path: str
    file_name: str
    object: str
    scene: str
    view: str
    lighting: str
    style: str
    prompt: str


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    difficulty: str
    query_type: str
    language: str
    text: str
    expected_ids: tuple[int, ...]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _image_id(path: Path) -> int:
    match = IMAGE_ID_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"圖片檔名必須以八位數 ID 開頭：{path.name}")
    return int(match.group(1))


def load_benchmark_images(folder: Path) -> list[BenchmarkImage]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"找不到評測圖片資料夾：{folder}")
    result: list[BenchmarkImage] = []
    errors: list[str] = []
    for image_path in sorted(folder.glob("*.png"), key=_image_id):
        try:
            with Image.open(image_path) as image:
                prompt_payload = image.info.get("prompt")
            if not prompt_payload:
                raise ValueError("缺少 PNG prompt metadata")
            graph = json.loads(prompt_payload)
            prompt = next(
                str(node.get("inputs", {}).get("text", "")).strip()
                for node in graph.values()
                if node.get("class_type") == "CLIPTextEncode"
                and str(node.get("inputs", {}).get("text", "")).strip()
            )
            parts = prompt.split(", ")
            if len(parts) != 6:
                raise ValueError(f"預期六個 prompt 欄位，實際為 {len(parts)}")
            object_name, scene, view, lighting, style, _detail = parts
            result.append(BenchmarkImage(
                _image_id(image_path), str(image_path), image_path.name,
                object_name, scene, view, lighting, style, prompt,
            ))
        except Exception as exc:
            errors.append(f"{image_path.name}: {exc}")
    if errors:
        preview = "\n".join(errors[:10])
        suffix = f"\n另有 {len(errors) - 10} 筆錯誤" if len(errors) > 10 else ""
        raise ValueError(f"無法建立評測標籤：\n{preview}{suffix}")
    if not result:
        raise ValueError(f"資料夾內沒有可評測的 PNG：{folder}")
    ids = [item.image_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("圖片 ID 重複，請確認檔名")
    return result


def build_queries(
    images: Sequence[BenchmarkImage], suite: str = "objective", languages: Sequence[str] = ("en",),
) -> list[BenchmarkQuery]:
    """Build the objective benchmark.

    ``core`` and ``full`` are retained as compatibility aliases for callers
    created before the subjective benchmark categories were retired.  All
    accepted suite names deliberately produce the same objective query set.
    """
    if suite not in SUPPORTED_SUITES:
        raise ValueError("suite 必須是 objective、core 或 full")
    invalid_languages = set(languages) - {"en", "zh", "mixed"}
    if invalid_languages or not languages or ("mixed" in languages and tuple(languages) != ("mixed",)):
        raise ValueError("languages 必須是 en、zh、mixed，或 en＋zh")

    specs: list[tuple[str, str, str, str, tuple[int, ...]]] = []

    def add_spec(difficulty: str, query_type: str, en: str, zh: str, members: Sequence[BenchmarkImage]):
        expected_ids = tuple(sorted({item.image_id for item in members}))
        if expected_ids:
            specs.append((difficulty, query_type, en, zh, expected_ids))

    objects = _unique(image.object for image in images)
    for object_name in objects:
        add_spec("基礎", "主體", object_name, OBJECT_ZH.get(object_name, object_name),
                 [image for image in images if image.object == object_name])
    for en, zh, object_name in ALIASES:
        add_spec("基礎", "同義詞", en, zh,
                 [image for image in images if image.object == object_name])
    for scene in _unique(image.scene for image in images):
        add_spec("中階", "場景", scene, SCENE_ZH.get(scene, scene),
                 [image for image in images if image.scene == scene])

    unexpected = {query_type for _difficulty, query_type, _en, _zh, _ids in specs} - OBJECTIVE_QUERY_TYPES
    if unexpected:
        raise RuntimeError(f"客觀題庫出現未允許類型：{', '.join(sorted(unexpected))}")

    queries: list[BenchmarkQuery] = []
    for number, (difficulty, query_type, en, zh, expected_ids) in enumerate(specs, 1):
        selected_languages = ("en" if number % 2 else "zh",) if tuple(languages) == ("mixed",) else languages
        for language in selected_languages:
            text = en if language == "en" else zh
            queries.append(BenchmarkQuery(
                f"Q{number:03d}-{language.upper()}", difficulty, query_type,
                language, text, expected_ids,
            ))
    return queries


def indexed_dataset_paths(images: Sequence[BenchmarkImage], asset_paths: Iterable[str]) -> tuple[int, list[str]]:
    indexed = {os.path.normcase(os.path.abspath(path)) for path in asset_paths}
    missing = [image.file_name for image in images if os.path.normcase(os.path.abspath(image.path)) not in indexed]
    return len(images) - len(missing), missing


def score_hits(
    query: BenchmarkQuery,
    hits: Sequence[SearchHit],
    images: Sequence[BenchmarkImage],
    top_k: int,
    latency_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    path_to_id = {os.path.normcase(os.path.abspath(image.path)): image.image_id for image in images}
    seen_paths: set[str] = set()
    ranked: list[tuple[SearchHit, int | None, bool]] = []
    for hit in hits:
        normalized = os.path.normcase(os.path.abspath(hit.asset.path))
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        result_id = path_to_id.get(normalized)
        ranked.append((hit, result_id, result_id in query.expected_ids if result_id is not None else False))

    expected = set(query.expected_ids)
    correct_all_ids = {result_id for _hit, result_id, correct in ranked if correct and result_id is not None}
    top = ranked[:top_k]
    correct_top_ids = {result_id for _hit, result_id, correct in top if correct and result_id is not None}
    returned_total = len(ranked)
    correct_total = len(correct_all_ids)
    correct_at_k = len(correct_top_ids)
    first_correct_rank = next((rank for rank, (_hit, _id, correct) in enumerate(top, 1) if correct), None)
    row = {
        "query_id": query.query_id,
        "difficulty": query.difficulty,
        "query_type": query.query_type,
        "language": query.language,
        "query": query.text,
        "expected_ids": ",".join(f"{item:03d}" for item in query.expected_ids),
        "relevant_total": len(expected),
        "returned_total": returned_total,
        "correct_total": correct_total,
        "precision_returned": correct_total / returned_total if returned_total else 0.0,
        "recall_returned": correct_total / len(expected) if expected else 0.0,
        "top_k": top_k,
        "correct_at_k": correct_at_k,
        "precision_at_k": correct_at_k / top_k,
        "recall_at_k": correct_at_k / len(expected) if expected else 0.0,
        "hit_at_1": int(bool(top and top[0][2])),
        "mrr_at_k": 1.0 / first_correct_rank if first_correct_rank else 0.0,
        "latency_seconds": latency_seconds,
        "top_result_ids": ",".join(f"{result_id:03d}" if result_id is not None else "外部" for _hit, result_id, _correct in top),
        "correct_ids": ",".join(f"{item:03d}" for item in sorted(correct_all_ids)),
        "missing_ids": ",".join(f"{item:03d}" for item in sorted(expected - correct_all_ids)),
        "error": "",
    }
    details = []
    for rank, (hit, result_id, correct) in enumerate(ranked, 1):
        details.append({
            "query_id": query.query_id,
            "query": query.text,
            "rank": rank,
            "file_name": Path(hit.asset.path).name,
            "image_id": f"{result_id:03d}" if result_id is not None else "",
            "in_benchmark_folder": int(result_id is not None),
            "correct": int(correct),
            "final_score": float(hit.score),
            "deterministic_score": float(hit.deterministic_score),
            "vlm_score": hit.vlm_score,
            "rerank_source": hit.rerank_source,
            "path": hit.asset.path,
        })
    return row, details


def run_benchmark(
    service,
    queries: Sequence[BenchmarkQuery],
    images: Sequence[BenchmarkImage],
    top_k: int,
    mode: SearchMode,
    progress: Callable[[int, int, dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    query_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for index, query in enumerate(queries, 1):
        started = time.perf_counter()
        try:
            hits = service.search(query.text, None, mode=mode)
            row, query_details = score_hits(
                query, hits, images, top_k, time.perf_counter() - started,
            )
            details.extend(query_details)
        except Exception as exc:
            row = {
                "query_id": query.query_id,
                "difficulty": query.difficulty,
                "query_type": query.query_type,
                "language": query.language,
                "query": query.text,
                "expected_ids": ",".join(f"{item:03d}" for item in query.expected_ids),
                "relevant_total": len(query.expected_ids),
                "returned_total": 0,
                "correct_total": 0,
                "precision_returned": 0.0,
                "recall_returned": 0.0,
                "top_k": top_k,
                "correct_at_k": 0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "hit_at_1": 0,
                "mrr_at_k": 0.0,
                "latency_seconds": time.perf_counter() - started,
                "top_result_ids": "",
                "correct_ids": "",
                "missing_ids": ",".join(f"{item:03d}" for item in query.expected_ids),
                "error": str(exc),
            }
        query_rows.append(row)
        if progress is not None:
            progress(index, len(queries), row)
    return query_rows, details


def _mean(rows: Sequence[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[position]


def aggregate_results(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if not row.get("error")]
    total_returned = sum(int(row["returned_total"]) for row in successful)
    total_correct = sum(int(row["correct_total"]) for row in successful)
    total_relevant = sum(int(row["relevant_total"]) for row in successful)
    latencies = [float(row["latency_seconds"]) for row in successful]

    def summary(group: Sequence[dict[str, object]]) -> dict[str, object]:
        returned = sum(int(row["returned_total"]) for row in group)
        correct = sum(int(row["correct_total"]) for row in group)
        relevant = sum(int(row["relevant_total"]) for row in group)
        return {
            "queries": len(group),
            "precision_macro": _mean(group, "precision_returned"),
            "recall_macro": _mean(group, "recall_at_k"),
            "precision_at_k_macro": _mean(group, "precision_at_k"),
            "hit_at_1": _mean(group, "hit_at_1"),
            "mrr_at_k": _mean(group, "mrr_at_k"),
            "precision_micro": correct / returned if returned else 0.0,
            "recall_micro": correct / relevant if relevant else 0.0,
        }

    by_type: dict[str, object] = {}
    for query_type in _unique(str(row["query_type"]) for row in successful):
        by_type[query_type] = summary([row for row in successful if row["query_type"] == query_type])
    by_language: dict[str, object] = {}
    for language in _unique(str(row["language"]) for row in successful):
        by_language[language] = summary([row for row in successful if row["language"] == language])
    overall = summary(successful)
    overall.update({
        "attempted_queries": len(rows),
        "failed_queries": len(rows) - len(successful),
        "latency_p50_seconds": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "total_returned": total_returned,
        "total_correct": total_correct,
        "total_relevant": total_relevant,
    })
    return {"overall": overall, "by_type": by_type, "by_language": by_language}


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


QUERY_FIELDS = (
    "query_id", "difficulty", "query_type", "language", "query", "expected_ids",
    "relevant_total", "returned_total", "correct_total", "precision_returned",
    "recall_returned", "top_k", "correct_at_k", "precision_at_k", "recall_at_k",
    "hit_at_1", "mrr_at_k", "latency_seconds", "top_result_ids", "correct_ids",
    "missing_ids", "error",
)

DETAIL_FIELDS = (
    "query_id", "query", "rank", "file_name", "image_id", "in_benchmark_folder",
    "correct", "final_score", "deterministic_score", "vlm_score", "rerank_source", "path",
)


def _pct(value: object) -> str:
    return f"{float(value):.1%}"


def _html_report(metadata: dict[str, object], aggregate: dict[str, object], rows: Sequence[dict[str, object]]) -> str:
    overall = aggregate["overall"]
    cards = [
        ("完成查詢", f"{overall['queries']} / {overall['attempted_queries']}"),
        ("平均 Precision", _pct(overall["precision_macro"])),
        (f"平均 Recall@{metadata['top_k']}", _pct(overall["recall_macro"])),
        ("Hit@1", _pct(overall["hit_at_1"])),
        ("MRR", _pct(overall["mrr_at_k"])),
        ("延遲 p95", f"{float(overall['latency_p95_seconds']):.1f}s"),
    ]
    card_html = "".join(
        f'<section class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    )
    type_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td>{data['queries']}</td>"
        f"<td>{_pct(data['precision_macro'])}</td>"
        f"<td>{_pct(data['recall_macro'])}</td>"
        f"<td>{_pct(data['hit_at_1'])}</td>"
        f"<td>{_pct(data['mrr_at_k'])}</td>"
        "</tr>"
        for name, data in aggregate["by_type"].items()
    )
    query_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['query_id']))}</td>"
        f"<td>{html.escape(str(row['query_type']))}</td>"
        f"<td>{html.escape(str(row['language']).upper())}</td>"
        f"<td class='query'>{html.escape(str(row['query']))}</td>"
        f"<td>{html.escape(str(row['expected_ids']))}</td>"
        f"<td>{html.escape(str(row['top_result_ids']))}</td>"
        f"<td class='{('good' if float(row['precision_returned']) >= .95 else 'warn' if float(row['precision_returned']) >= .8 else 'bad')}'>{_pct(row['precision_returned'])}</td>"
        f"<td>{_pct(row['recall_at_k'])}</td>"
        f"<td>{_pct(row['hit_at_1'])}</td>"
        f"<td>{float(row['latency_seconds']):.1f}s</td>"
        f"<td class='error'>{html.escape(str(row['error']))}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Search 準確度報告</title>
<style>
:root{{--navy:#17324d;--teal:#2f7f7a;--bg:#f4f7f9;--line:#d0d5dd;--green:#e6f4ea;--yellow:#fff4cc;--red:#fce8e6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:#172b3a;font:14px/1.5 "Segoe UI","Noto Sans TC",sans-serif}}
header{{background:var(--navy);color:white;padding:28px 5vw}} header h1{{margin:0 0 6px;font-size:28px}} header p{{margin:0;color:#dbe7f1}}
main{{max-width:1500px;margin:auto;padding:24px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:22px}}
.card{{background:white;border-radius:12px;padding:16px;border:1px solid #e4e7ec}} .card span{{display:block;color:#667085}} .card strong{{display:block;font-size:26px;color:var(--teal);margin-top:6px}}
.panel{{background:white;border:1px solid #e4e7ec;border-radius:12px;margin:18px 0;overflow:auto}} .panel h2{{font-size:18px;margin:0;padding:16px 18px;border-bottom:1px solid var(--line)}}
.note{{background:#e8f3f2;border-left:5px solid var(--teal);padding:12px 16px;margin:0 0 18px}}
table{{border-collapse:collapse;width:100%;min-width:900px}} th{{position:sticky;top:0;background:var(--teal);color:white;text-align:left}} th,td{{padding:9px 10px;border-bottom:1px solid #e8eaed;vertical-align:top}} tr:hover td{{background:#f8fafb}} .query{{font-weight:600}} .good{{background:var(--green);font-weight:700}} .warn{{background:var(--yellow);font-weight:700}} .bad{{background:var(--red);font-weight:700}} .error{{color:#b42318;max-width:260px}}
</style></head><body>
<header><h1>Visual Search 準確度報告</h1><p>{html.escape(str(metadata['created_at']))} · {html.escape(str(metadata['suite']))} · {html.escape(str(metadata['languages']))} · {html.escape(str(metadata['mode']))} · 門檻 {metadata['threshold']}</p></header>
<main><div class="note">Precision（回傳）與你原本的算法相同：正確張數 ÷ 找到張數。標準答案來自測試 PNG 的 ComfyUI prompt metadata。</div>
<div class="cards">{card_html}</div>
<section class="panel"><h2>依查詢類型</h2><table><thead><tr><th>類型</th><th>查詢數</th><th>Precision</th><th>Recall@{metadata['top_k']}</th><th>Hit@1</th><th>MRR</th></tr></thead><tbody>{type_rows}</tbody></table></section>
<section class="panel"><h2>每個查詢明細</h2><table><thead><tr><th>ID</th><th>類型</th><th>語言</th><th>查詢</th><th>標準答案</th><th>Top {metadata['top_k']} 結果</th><th>Precision</th><th>Recall</th><th>Hit@1</th><th>耗時</th><th>錯誤</th></tr></thead><tbody>{query_rows}</tbody></table></section>
</main></body></html>"""


def write_reports(
    output_root: Path,
    query_rows: Sequence[dict[str, object]],
    details: Sequence[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir = output_root / timestamp
    suffix = 2
    while report_dir.exists():
        report_dir = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    report_dir.mkdir(parents=True)
    aggregate = aggregate_results(query_rows)
    payload = {"metadata": metadata, "aggregate": aggregate, "queries": list(query_rows)}
    json_path = report_dir / "accuracy-report.json"
    query_csv_path = report_dir / "accuracy-queries.csv"
    detail_csv_path = report_dir / "accuracy-results.csv"
    html_path = report_dir / "accuracy-report.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(query_csv_path, query_rows, QUERY_FIELDS)
    _write_csv(detail_csv_path, details, DETAIL_FIELDS)
    html_path.write_text(_html_report(metadata, aggregate, query_rows), encoding="utf-8")
    latest = output_root / "latest-accuracy-report.html"
    latest.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "directory": report_dir,
        "html": html_path,
        "json": json_path,
        "queries_csv": query_csv_path,
        "results_csv": detail_csv_path,
    }


def benchmark_metadata(
    *, image_folder: Path, suite: str, languages: Sequence[str], mode: SearchMode,
    threshold: int, top_k: int, image_count: int,
) -> dict[str, object]:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_folder": str(image_folder.resolve()),
        "suite": suite,
        "languages": ",".join(languages),
        "mode": mode.value,
        "threshold": threshold,
        "top_k": top_k,
        "image_count": image_count,
    }
