# Visual Search 1.0 發布清單

程式、Windows runtime 與 installer 已完成。正式交給其他使用者前，還需要由 repository 擁有者上傳自有模型並固定來源；腳本不會保存 `HF_TOKEN`。

## 1. 建立並上傳兩個核心模型 repo

在 Hugging Face 建立兩個 public model repositories，例如：

```text
你的帳號/visual-search-siglip2-naflex-openvino
你的帳號/visual-search-ppocrv6-openvino
```

驗證並上傳既有 OpenVINO 模型：

```powershell
$env:HF_TOKEN = "hf_..."

.\.venv\Scripts\python.exe scripts\prepare-hf-upload.py `
  --folder release-models\siglip2-so400m-patch16-naflex-openvino-fp16 `
  --repo 你的帳號/visual-search-siglip2-naflex-openvino

.\.venv\Scripts\hf.exe upload 你的帳號/visual-search-siglip2-naflex-openvino `
  release-models\siglip2-so400m-patch16-naflex-openvino-fp16 . --repo-type model

.\.venv\Scripts\python.exe scripts\finalize-model-source.py `
  --section model --repo 你的帳號/visual-search-siglip2-naflex-openvino

.\.venv\Scripts\python.exe scripts\prepare-hf-upload.py `
  --folder release-models\pp-ocrv6-medium-openvino-fp16 `
  --repo 你的帳號/visual-search-ppocrv6-openvino

.\.venv\Scripts\hf.exe upload 你的帳號/visual-search-ppocrv6-openvino `
  release-models\pp-ocrv6-medium-openvino-fp16 . --repo-type model

.\.venv\Scripts\python.exe scripts\finalize-model-source.py `
  --section ocr_model --repo 你的帳號/visual-search-ppocrv6-openvino
```

`finalize-model-source.py` 會把遠端 immutable commit、每檔大小及 SHA-256 寫入 `src/visual_search/resources/models.json`。在這兩段完成前，本機開發版會使用已安裝模型，但新使用者的乾淨安裝版無法自動下載 SigLIP2／OCR。

Qwen3.5 9B INT8 與 INT4 已鎖定官方 OpenVINO repositories。YOLOE-26S／M Prompt-Free 必須先執行 `convert-yoloe26-openvino-wsl.py`，自行建立兩個 Hugging Face repositories，上傳後分別以 `--section yoloe_s` 與 `--section yoloe_m` 固定來源。ONNX 驗證檔不得上傳或放進 installer。

## 2. 建立 GitHub repo 並建置公開版

```powershell
$env:VISUAL_SEARCH_GITHUB_REPOSITORY = "你的帳號/visual-search"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

這會將 repo 路徑與本次 installer SHA 寫入 `dist/install.ps1`。確認其中沒有 `REPLACE_WITH`：

```powershell
Select-String -Path dist\install.ps1 -Pattern 'REPLACE_WITH'
```

## 3. GitHub Release v1.0.0

上傳：

```text
dist/VisualSearch-Setup.exe
dist/VisualSearch-Setup.exe.sha256
dist/install.ps1
dist/sbom.json
```

同時公開完整 AGPL-3.0 對應原始碼、`LICENSE`、`THIRD_PARTY_NOTICES.md`、建置腳本與鎖定 manifest。

## 4. 發布前驗證

```powershell
.\dist\VisualSearch\VisualSearch.exe --diagnose
Get-Content "$env:LOCALAPPDATA\VisualSearch\logs\latest-diagnostics.json"
```

在乾淨 Windows 10／11 帳號驗證安裝、四個模型下載、索引、中文／英文搜尋、VLM Top 30、解除安裝與重新安裝。正式簽章若暫時沒有商業憑證，SHA-256 仍必須發布；未簽章 installer 會觸發 SmartScreen 提示。
