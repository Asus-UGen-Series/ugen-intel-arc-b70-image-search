# Visual Search

[English](README.md) | 繁體中文

Visual Search 是重視隱私的 Windows 本機圖片搜尋應用程式，可用自然語言描述、圖片內文字、檔名、物件與日期尋找圖片。它會為使用者選取的資料夾建立可續跑的增量索引，原始圖片只保留在電腦上，不會上傳。

## 功能

- 支援英文與繁體中文的自然語言圖片搜尋。
- 結合 OCR、精確檔名與識別碼、物件資訊、EXIF metadata 及語意圖片向量。
- 使用 SigLIP2 全圖與區域向量處理大型圖片及小物件。
- 可選用 Qwen3.5 視覺審核，提供「智慧完整」與「徹底搜尋」模式。
- 遞迴監看資料夾、增量重建索引，並可在中斷後安全接續。
- 使用本機 SQLite FTS5 與 USearch，不需要外部資料庫。
- 提供 Windows 安裝程式，以及獨立的開發者 Debug UI。

## 系統需求

- 64 位元 Windows 10 或 Windows 11。
- OpenVINO 支援的 Intel GPU；參考系統為 Intel Arc Pro B70 32 GB。
- 第一次啟動下載模型時需要網路連線。
- 模型、縮圖與索引需要額外磁碟空間。四個核心 OpenVINO 模型合計約 2.43 GB；選配 Qwen INT4 約需 6.13 GB，INT8 約需 9.46 GB。

## 安裝

1. 從本 GitHub Repo 的最新 Release 下載 `VisualSearch-Setup.exe` 與 `VisualSearch-Setup.exe.sha256`。
2. 執行前驗證安裝程式：

   ```powershell
   Get-FileHash .\VisualSearch-Setup.exe -Algorithm SHA256
   ```

3. 執行 `VisualSearch-Setup.exe`，再從開始功能表啟動 Visual Search。

安裝程式以目前使用者身分安裝，不需要系統管理員權限。它已包含 Python、Qt、OpenVINO、SQLite 與索引 runtime；使用者不需要另外安裝 Python、Winget、Git、Docker 或資料庫。

第一次啟動時，程式會下載並驗證核心模型，接著可選擇 Qwen3.5 9B INT8、Qwen3.5 9B INT4、本機 OpenVINO 模型資料夾或 ZIP，或停用 VLM。模型與應用程式資料預設存放於 `%LOCALAPPDATA%\VisualSearch`。

## 搜尋架構

| 元件 | 用途 | 發布來源 |
| --- | --- | --- |
| SigLIP2 SO400M NaFlex FP16 | 全圖與區域圖片／文字向量 | [`WarrenYD/visual-search-openvino-models`](https://huggingface.co/WarrenYD/visual-search-openvino-models)，固定於 [`models.json`](src/visual_search/resources/models.json) |
| PP-OCRv6 Medium FP16 | 高解析度自適應 OCR | [`WarrenYD/visual-search-openvino-models`](https://huggingface.co/WarrenYD/visual-search-openvino-models)，固定於 [`models.json`](src/visual_search/resources/models.json) |
| YOLOE-26S/M Prompt-Free FP16 | 物件排序軟加分，不作 hard filter | [`WarrenYD/visual-search-openvino-models`](https://huggingface.co/WarrenYD/visual-search-openvino-models)，固定於 [`models.json`](src/visual_search/resources/models.json) |
| Qwen3.5 9B INT8/INT4 | 選配的候選圖片視覺審核 | 官方 OpenVINO Repo，固定於 [`models.json`](src/visual_search/resources/models.json) |
| SQLite FTS5 與 USearch | Metadata、OCR、處理狀態與向量召回 | 隨應用程式提供 |

所有 GPU 工作共用一個優先順序排程器：查詢向量、VLM 審核、背景索引、暖機與診斷。搜尋時會暫停背景索引，完成後自動接續。

「智慧完整」會動態擴張語意召回並審核候選集合中的每一張圖片；「徹底搜尋」會審核符合 metadata 條件的整個圖片庫。結果必須達到所選的 60、70 或 85 分門檻，且不能被標記為 hard failure。精確檔名、OCR 片語、型號與程式碼識別碼命中不會被 VLM 誤刪。

## 隱私與本機資料

- 原始圖片與搜尋文字不會上傳。
- VLM 判定會在本機快取 30 天；快取不保存查詢文字，也不另外複製圖片。
- 移除監看資料夾時，只會刪除其 Visual Search 索引資料，不會刪除原始檔案。
- 磁碟暫時離線時，不會因此移除整個索引。

## 開發

支援 Python 3.11 至 3.13。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -e ".[test,build]"
.\scripts\run-dev.cmd
```

執行測試與診斷：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe launcher.py --diagnose
.\.venv\Scripts\python.exe launcher.py --export-logs
.\.venv\Scripts\python.exe launcher.py --resume-index
```

`quick-test.cmd` 提供互動式繁體中文開發者選單，也可直接傳入 `test`、`diagnose`、`repair`、`app`、`evaluate` 或 `all`。`quick-test.cmd app` 會啟用 Debug UI，顯示每張圖片評分、召回漏斗診斷、索引覆蓋、實際索引路徑及最近的 VLM 活動。Frozen 正式版不會顯示或保存開發者 trace。

## 客觀 test100 評測

[`benchmarks/test100/images`](benchmarks/test100/images) 包含 100 張保留 ComfyUI prompt metadata 的 PNG，並以此作為客觀標準答案。評測只產生主體、同義詞與場景題：英文 41 題、繁體中文 41 題，兩種語言各跑一次共 82 題。

不載入模型，只驗證資料集：

```powershell
.\.venv\Scripts\python.exe .\evaluate_test100.py --check
```

執行建議評測設定：

```powershell
.\.venv\Scripts\python.exe .\evaluate_test100.py --yes --language both `
  --mode smart_complete --threshold 85 --top-k 10 --no-open-report
```

CLI 也支援單一語言、中英混合 41 題、徹底搜尋、Top 5／10／20，以及自訂圖片或報告目錄。

版本控制中的 [82 題評測報告](benchmarks/test100/accuracy-report.html) 使用智慧完整、85 分門檻及 Top 10，結果如下：

| 指標 | 結果 |
| --- | ---: |
| 平均 Precision | 90.8% |
| 平均 Recall@10 | 90.8% |
| Hit@1 | 100.0% |
| MRR | 100.0% |
| p95 延遲 | 15.8 秒 |

## 建置

Inno Setup 6 只需安裝於開發電腦。

```powershell
.\scripts\build.ps1
```

正式建置必須設定 `VISUAL_SEARCH_GITHUB_REPOSITORY=owner/repository`。發布檢查會拒絕缺少模型 Repo、可變或無效 revision、錯誤檔案大小及 SHA-256 不一致的內容。

建置輸出：

```text
dist/VisualSearch-Setup.exe
dist/VisualSearch-Setup.exe.sha256
dist/install.ps1
dist/sbom.json
```

## 常見問題

- 模型下載失敗時，請確認網路連線後重試；未完成的下載會自動接續。
- 評測回報缺少圖片或 metadata 時，請確認 100 張 PNG 都存在，且仍保留 ComfyUI prompt metadata。
- 評測回報索引未完成時，先將 `benchmarks/test100/images` 加入 Visual Search，等待背景分析完成。
- GPU 記憶體不足時，關閉其他 Visual Search 或評測程序、改用 Qwen INT4，再接續索引。
- 未簽章的開發版安裝程式可能觸發 Windows SmartScreen；繼續前請先核對發布的 SHA-256。

## 授權

應用程式原始碼採 AGPL-3.0-only。第三方元件與模型保留各自的授權條款。詳見 [`LICENSE`](LICENSE)、[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 與建置產生的 `dist/sbom.json`。

作者：Warren Chu
