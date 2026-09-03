# Visual Search 1.0 發布清單

程式、Windows runtime 與 installer 可在本機建置。正式交給其他使用者前，必須先發布自製模型並固定所有下載來源；腳本不會保存 `HF_TOKEN`。

## 1. 驗證整合模型 staging

四套自製 OpenVINO 模型必須位於：

```text
release-models/visual-search-openvino-models/
  siglip2-so400m-patch16-naflex-openvino-fp16/
  pp-ocrv6-medium-openvino-fp16/
  yoloe-26s-pf-openvino-fp16/
  yoloe-26m-pf-openvino-fp16/
```

產生可重現的大小與 SHA-256 清單：

```powershell
.\.venv\Scripts\python.exe scripts\prepare-hf-upload.py `
  --folder release-models\visual-search-openvino-models `
  --repo 你的帳號/visual-search-openvino-models
```

## 2. 建立一個公開 Hugging Face model repo

```powershell
$env:HF_TOKEN = "hf_..."
.\.venv\Scripts\python.exe scripts\publish-model.py --bundle `
  --repo 你的帳號/visual-search-openvino-models
```

此命令會以單一 commit 上傳四套自製模型，並將同一個 immutable revision、遠端子路徑、檔案大小及 SHA-256 寫入 `src/visual_search/resources/models.json`。Qwen3.5 INT8／INT4 維持官方 OpenVINO repositories，不得重複上傳。

再次執行正式檢查：

```powershell
.\.venv\Scripts\python.exe scripts\check-release-models.py
```

任何一個 section 未設定都必須阻止發布。

## 3. 建立 GitHub repo 與正式安裝包

```powershell
$env:VISUAL_SEARCH_GITHUB_REPOSITORY = "你的帳號/visual-search"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

確認 `dist/install.ps1` 不含 placeholder：

```powershell
Select-String -Path dist\install.ps1 -Pattern 'REPLACE_WITH'
```

## 4. GitHub Release v1.0.0

上傳：

```text
dist/VisualSearch-Setup.exe
dist/VisualSearch-Setup.exe.sha256
dist/install.ps1
dist/sbom.json
```

同時公開 AGPL-3.0 對應原始碼、`tests/`、`quick-test.cmd`、test100、單一歷史評測報告、授權、建置腳本與鎖定 manifest。不得提交 `release-models/`、`.venv/`、`build/`、`dist/` 或其他本機產物。

## 5. 發布前驗收

- 執行 pytest、`evaluate_test100.py --check` 與 public-repo scan。
- 在含空白路徑下安裝、啟動、執行診斷及解除安裝。
- 在乾淨 Windows 10／11 帳號驗證四套自製模型及選定 Qwen VLM 的首次下載、大小和 SHA-256。
- 加入 test100，等待 100 張完成索引，再執行完整 82 查詢。
- 正式簽章若暫時沒有商業憑證，仍必須發布 SHA-256，並記錄 Windows SmartScreen 提示。
