# Visual Search

English | [繁體中文](README.zh-TW.md)

Visual Search is a privacy-first Windows desktop application for finding local images with natural-language descriptions, text inside images, file names, objects, and dates. It builds a resumable incremental index for user-selected folders. Original images stay on the computer and are never uploaded.

## Features

- Natural-language image retrieval in English and Traditional Chinese.
- OCR, exact file-name and identifier matching, object evidence, EXIF metadata, and semantic image embeddings.
- Global and region-level SigLIP2 embeddings for large images and small objects.
- Optional Qwen3.5 visual verification with Smart Complete and Exhaustive Search modes.
- Recursive folder monitoring, incremental re-indexing, and safe resume after interruption.
- Local SQLite FTS5 and USearch indexes with no external database.
- A packaged Windows installer and a separate developer Debug UI.

## Requirements

- 64-bit Windows 10 or Windows 11.
- An OpenVINO-compatible Intel GPU. The reference system is Intel Arc Pro B70 32 GB.
- Internet access during the first launch to download model files.
- Additional disk space for models, thumbnails, and indexes. The four core OpenVINO models total approximately 2.43 GB; the optional Qwen model requires approximately 6.13 GB for INT4 or 9.46 GB for INT8.

## Installation

1. Download `VisualSearch-Setup.exe` and `VisualSearch-Setup.exe.sha256` from this repository's latest GitHub Release.
2. Verify the installer before running it:

   ```powershell
   Get-FileHash .\VisualSearch-Setup.exe -Algorithm SHA256
   ```

3. Run `VisualSearch-Setup.exe` and launch Visual Search from the Start menu.

The installer is per-user and does not require administrator privileges. Python, Qt, OpenVINO, SQLite, and the index runtime are included. Python, Winget, Git, Docker, and a separate database are not required.

On first launch, the application downloads and verifies the core models. It then offers Qwen3.5 9B INT8, Qwen3.5 9B INT4, a local OpenVINO model folder or ZIP, or VLM-disabled operation. Models and application data are stored under `%LOCALAPPDATA%\VisualSearch`.

## Search architecture

| Component | Purpose | Distribution |
| --- | --- | --- |
| SigLIP2 SO400M NaFlex FP16 | Global and region image/text embeddings | [`WarrenYD/visual-search-openvino-models`](https://huggingface.co/WarrenYD/visual-search-openvino-models), pinned in [`models.json`](src/visual_search/resources/models.json) |
| PP-OCRv6 Medium FP16 | High-resolution adaptive OCR | [`WarrenYD/visual-search-openvino-models`](https://huggingface.co/WarrenYD/visual-search-openvino-models), pinned in [`models.json`](src/visual_search/resources/models.json) |
| YOLOE-26S/M Prompt-Free FP16 | Soft object-ranking evidence; never a hard filter | [`WarrenYD/visual-search-openvino-models`](https://huggingface.co/WarrenYD/visual-search-openvino-models), pinned in [`models.json`](src/visual_search/resources/models.json) |
| Qwen3.5 9B INT8/INT4 | Optional visual verification of candidates | Official OpenVINO repositories pinned in [`models.json`](src/visual_search/resources/models.json) |
| SQLite FTS5 and USearch | Metadata, OCR, pipeline state, and vector retrieval | Included with the application |

All GPU work uses one priority scheduler: query embeddings, VLM verification, background indexing, then warm-up and diagnostics. Background indexing pauses while a search is active and resumes automatically.

Smart Complete expands semantic recall dynamically and verifies every candidate in the selected set. Exhaustive Search verifies the full metadata-eligible library. Results must meet the selected 60, 70, or 85 score threshold and must not be marked as a hard failure. Exact file-name, OCR phrase, model number, and code-identifier matches are protected from accidental VLM removal.

## Privacy and local data

- Original images and search text are not uploaded.
- VLM verdicts are cached locally for 30 days; cached entries do not contain the query text or an extra image copy.
- Removing a watched folder deletes only its Visual Search index data. Original files are not deleted.
- A temporarily unavailable drive does not cause its complete index to be removed.

## Development

Python 3.11 through 3.13 is supported.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -e ".[test,build]"
.\scripts\run-dev.cmd
```

Run tests and diagnostics:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe launcher.py --diagnose
.\.venv\Scripts\python.exe launcher.py --export-logs
.\.venv\Scripts\python.exe launcher.py --resume-index
```

`quick-test.cmd` provides an interactive Traditional Chinese developer menu and also accepts `test`, `diagnose`, `repair`, `app`, `evaluate`, or `all` directly. `quick-test.cmd app` enables the Debug UI with per-image scores, retrieval-funnel diagnostics, index coverage, actual index paths, and recent VLM activity. Frozen release builds do not expose or store developer traces.

## Objective test100 benchmark

[`benchmarks/test100/images`](benchmarks/test100/images) contains 100 PNG files with ComfyUI prompt metadata used as objective ground truth. The evaluator generates only subject, synonym, and scene queries: 41 English queries and 41 Traditional Chinese queries, or 82 when both languages run.

Validate the dataset without loading models:

```powershell
.\.venv\Scripts\python.exe .\evaluate_test100.py --check
```

Run the recommended benchmark:

```powershell
.\.venv\Scripts\python.exe .\evaluate_test100.py --yes --language both `
  --mode smart_complete --threshold 85 --top-k 10 --no-open-report
```

The CLI also supports one language, a 41-query mixed-language run, Exhaustive Search, Top 5/10/20, and custom image or report directories.

The versioned [82-query report](benchmarks/test100/accuracy-report.html) records the following results for Smart Complete, threshold 85, and Top 10:

| Metric | Result |
| --- | ---: |
| Mean Precision | 90.8% |
| Mean Recall@10 | 90.8% |
| Hit@1 | 100.0% |
| MRR | 100.0% |
| p95 latency | 15.8 s |

## Build

Inno Setup 6 is required only on the development machine.

```powershell
.\scripts\build.ps1
```

Production builds must set `VISUAL_SEARCH_GITHUB_REPOSITORY=owner/repository`. The release check rejects missing model repositories, mutable or invalid revisions, incorrect file sizes, and SHA-256 mismatches.

Build output:

```text
dist/VisualSearch-Setup.exe
dist/VisualSearch-Setup.exe.sha256
dist/install.ps1
dist/sbom.json
```

## Troubleshooting

- If a model download fails, confirm network access and retry; partial downloads resume automatically.
- If evaluation reports missing images or metadata, verify that all 100 benchmark PNG files are present and retain their ComfyUI prompt metadata.
- If evaluation reports an incomplete index, add `benchmarks/test100/images` to Visual Search and wait for background analysis to finish.
- If GPU memory is exhausted, close other Visual Search or evaluation processes, select Qwen INT4, and resume indexing.
- An unsigned development installer may trigger Windows SmartScreen. Verify its published SHA-256 before continuing.

## License

The application source is licensed under AGPL-3.0-only. Third-party components and model artifacts retain their respective licenses. See [`LICENSE`](LICENSE), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and the generated `dist/sbom.json`.

Author: Warren Chu
