# Visual Search 1.0

Visual Search 是完全在本機執行的 Windows 圖片搜尋工具。使用者選擇圖片資料夾後，程式會建立可續跑的增量索引，並以自然語言、圖片內文字、檔名、物件與日期搜尋。原始圖片不會上傳。

正式目標是 Windows 10/11 64-bit 與 Intel Arc Pro B70 32 GB。安裝程式包含 Python、Qt、OpenVINO 與索引元件，不要求使用者另外安裝 Python、Winget、Git、Docker 或資料庫。

## 推論與資料架構

- SigLIP2 SO400M NaFlex OpenVINO：圖片與文字向量。
- PP-OCRv6 Medium OpenVINO：高解析度自適應分塊 OCR。
- YOLOE-26S／M Prompt-Free FP16 OpenVINO：固定開放詞彙的物件偵測，只作軟加分。
- SigLIP2 區域索引：每張高解析圖片建立最多六個類別無關局部向量，補足 YOLOE 詞彙之外的小物件。
- Qwen3.5 9B INT8／INT4 OpenVINO：選配的最終 Top 30 視覺審核。
- SQLite FTS5：路徑、EXIF、OCR、物件與階段狀態。
- USearch：向量近鄰索引；SQLite 保留向量備份。

所有 GPU 工作經同一個排程器執行：查詢文字向量 P0、VLM 審核 P1、背景索引 P2、暖機／診斷 P3。搜尋時不再派發新的背景 batch，完成後自動續跑。

VLM 只接收使用者原句、整理出的可見條件、圖片，以及最多兩行必要 OCR；不接收 YOLO label、座標、SigLIP 分數、RRF 名次或 pipeline log。它只能輸出：

```json
{"match": 91, "hard_fail": false}
```

無效 JSON、逾時、取消、OOM 或模型未安裝時，整次查詢回退 deterministic Top 10，不混用不完整的 VLM 結果。精確檔名、OCR 片語、型號與程式碼 identifier 命中不會被 VLM 刪除。

## 開發版執行

```powershell
git clone https://github.com/OWNER/visual-search.git
cd visual-search
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -e ".[test,build]"
.\scripts\run-dev.cmd
```

第一次啟動會檢查核心模型，並讓使用者選擇：

1. Qwen3.5 9B INT8（推薦，約 9.46 GB）。
2. Qwen3.5 9B INT4（快速，約 6.13 GB）。
3. 從本機 OpenVINO 模型資料夾或 ZIP 匯入。
4. 稍後安裝／停用 VLM。

模型與資料預設存放在 `%LOCALAPPDATA%\VisualSearch`。切換 VLM 不必重建圖片索引。

## 測試與診斷

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe launcher.py --diagnose
.\.venv\Scripts\python.exe launcher.py --export-logs
.\.venv\Scripts\python.exe launcher.py --reset-index
```

診斷輸出包含 OpenVINO runtime、實際執行裝置、SigLIP、OCR、YOLOE、VLM 的載入狀態、區域向量數、模型來源與 GPU 排程狀態。

## Qwen3.5 OpenVINO 轉換（WSL）

INT8 使用 `INT8_ASYM`、per-column（group size -1）；INT4 使用 `INT4_ASYM`、group size 128、ratio 1.0：

```bash
cd "/mnt/c/path/to/visual-search"
python3 scripts/convert-qwen35-openvino-wsl.py --weight-format int8
python3 scripts/convert-qwen35-openvino-wsl.py --weight-format int4
```

輸出位置：

```text
artifacts/hf/Qwen3.5-9B-int8-ov
artifacts/hf/Qwen3.5-9B-int4-ov
```

每個資料夾包含 OpenVINO IR、tokenizer／processor、README、授權資訊、SHA-256 manifest 與 Windows B70 smoke-test 指令。

## 上傳 Hugging Face

先自行建立 Hugging Face model repo，token 只放在 `HF_TOKEN` 環境變數：

```powershell
$env:HF_TOKEN = "hf_..."
.\.venv\Scripts\python.exe scripts\prepare-hf-upload.py --folder artifacts\hf\Qwen3.5-9B-int8-ov
.\.venv\Scripts\python.exe scripts\finalize-model-source.py --section vlm_int8 --repo 使用者名稱/Qwen3.5-9B-int8-ov
```

`finalize-model-source.py` 會取得不可變 commit revision、重新驗證遠端每個檔案的大小與 SHA-256，並更新 production manifest。不要在終端使用者電腦轉換 Safetensors，也不要執行 Hugging Face repository 內的自訂 Python。

SigLIP2 與 PP-OCRv6 的 WSL 轉換：

```bash
python3 scripts/convert-siglip2-openvino-wsl.py --force --install-local
python3 scripts/convert-paddleocr-openvino-wsl.py --force --install-local
```

YOLOE-26 Prompt-Free 轉換。Windows 可直接執行推薦的 S 模型轉換；依賴只會裝在被 Git 排除的 `artifacts/conversion-env-yoloe26`，不會污染應用程式 runtime：

```cmd
.\scripts\convert-yoloe26-openvino-windows.cmd
```

完成後會同時輸出至 `artifacts\hf\yoloe-26s-pf-openvino-fp16`，並安裝至 `%LOCALAPPDATA%\VisualSearch\models\yoloe-26s-pf-openvino-fp16`。重新啟動 Visual Search 即可載入。

也可在 WSL 轉換 S／M：

```bash
cd "/mnt/c/path/to/visual-search"
python3 scripts/convert-yoloe26-openvino-wsl.py --scale both --install-local
```

輸出位於 `artifacts/hf/yoloe-26s-pf-openvino-fp16` 與 `artifacts/hf/yoloe-26m-pf-openvino-fp16`。ONNX 只作轉換結果比對，預設驗證後刪除，不會進入 runtime 或安裝包。上傳後分別以 `--section yoloe_s`、`--section yoloe_m` 固定 Hugging Face revision。

### 開發者搜尋除錯

維護者的本機 checkout 可用根目錄 `quick-test.cmd` 啟動開發者模式：

```cmd
quick-test.cmd app
```

這個入口會設定 `VISUAL_SEARCH_DEVELOPER_MODE=1`，並顯示正式版沒有的「搜尋除錯（開發者）」頁。頁面會保存最近一次查詢的本機 trace，包含：

- 查詢類型與實際套用的 SigLIP 全圖、SigLIP 局部、OCR／檔名、YOLOE、Metadata 權重。
- 各通道套用 hard filter 前後的候選數、每張圖的原始相似度、名次、RRF 分數與加權貢獻。
- exact filename／OCR phrase／identifier bonus、deterministic 分數、VLM 分數、`hard_fail` 與最後分數。
- ready／partial／error 數量，以及 embedding、region、OCR、object 各索引階段的覆蓋率。
- 指定一個預期檔名後，判斷它是未進召回 Top 60，還是進入候選後被排名／VLM 移除。

評分公式為：

```text
RRF(rank) = 61 / (60 + rank)
deterministic = min(1, 0.75 * sum(channel_weight * RRF) + bonus)
final = 0.70 * VLM + 0.30 * deterministic   # 只有啟用 VLM 時
```

若本機沒有 `quick-test.cmd`，也可以在 CMD 暫時啟用：

```cmd
set VISUAL_SEARCH_DEVELOPER_MODE=1
scripts\run-dev.cmd
```

正式 `VisualSearch-Setup.exe` 為 frozen build，即使設定環境變數也不會顯示或保存 developer trace。`quick-test.cmd` 與 `tests/` 依產品需求只留維護者本機，不提交 Git。

## 建置 Windows 安裝程式

開發機需安裝 Inno Setup 6。終端使用者不需要安裝它。

```powershell
$env:VISUAL_SEARCH_GITHUB_REPOSITORY = "你的帳號/visual-search"
.\scripts\build.ps1
```

若尚未建立 GitHub repo，可省略環境變數先建立本機測試版；`dist/install.ps1` 會保留明確 placeholder 並拒絕下載，避免誤連到錯誤來源。正式發布前必須帶入 `owner/repository` 後重建。

成功後輸出：

```text
dist/VisualSearch-Setup.exe
dist/VisualSearch-Setup.exe.sha256
dist/install.ps1
dist/sbom.json
```

安裝器為目前 Windows 使用者安裝，不要求系統管理員權限。模型在第一次啟動時以斷點續傳下載，驗證大小與 SHA-256 後才原子切換。

## 發行驗收

- 在 Intel Arc B70 執行 `--diagnose`，確認四個模型的 `execution_devices` 是 Intel GPU。
- 使用固定資料集測量 INT8／INT4 Top 1、Top 3、Top 10、hard-fail 誤判率與 Top 30 p50／p95。
- 驗證全圖與區域 SigLIP 各自可召回候選；YOLOE 結果不得作 hard filter。
- INT8 Top 30 暖機 p95 必須不超過 60 秒；若 448px 超時，依序測試 384／336，品質退化超過 2% 就阻止發布。
- 在沒有 Python、Winget、Git、Docker 的乾淨 Windows 10／11 測試安裝、首次啟動、下載、索引、搜尋、解除安裝與重新安裝。

## 授權

應用程式原始碼採 AGPL-3.0-only。第三方元件與模型各自依其授權發布；詳見 `THIRD_PARTY_NOTICES.md`、`LICENSE` 與建置產生的 `dist/sbom.json`。
