# Fresh-install acceptance

Use a disposable Windows 10/11 VM for installation and a test PC with the target Intel GPU for hardware acceptance. A source test or a successful build is not clean-machine acceptance.

## Repeatable checks

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check-release-models.py
.\scripts\check-public-repo.ps1
.\scripts\build.ps1
.\scripts\verify-packaged-app.ps1
```

The packaged check uses `VISUAL_SEARCH_DATA_DIR` to create a separate empty temporary library. It retains diagnostics without changing the normal library. It checks the bundled runtime, not GPU inference.

## End-user checklist

Record OS, driver, installer hash, model revisions, and observed outcomes for every step.

| Step | Expected behavior |
| --- | --- |
| Download installer | Published SHA-256 matches; release link works without credentials |
| Install as standard user | No Python, Winget, Git, Docker or administrator setup required |
| Shortcuts and uninstall entry | Start-menu entry and optional desktop shortcut launch the installed EXE |
| First launch | Model preparation identifies local verification and missing downloads |
| Disconnect during download | Error and retry available; partial files resume |
| Existing models without marker | Hash-valid files reused without network; inference tested before marker creation |
| Low disk space | Check only space needed for missing files; preserve existing files |
| VLM INT8, INT4, local or off | Choice survives restart; invalid imports preserve the previous model |
| Unicode picture folder | Index progress and stage errors visible; damaged images handled |
| OCR and natural-language search | Correct candidates retrieved; actual VLM inference distinguished from cache |
| Search while indexing | Interactive progress visible; background indexing resumes |
| Replace/cancel query | Old query cannot overwrite new results |
| Close and restart | Completed stages reused; interrupted stages resume |
| Add/modify/move/delete image | Watcher and periodic reconciliation update the affected records |
| Offline drive | Existing index retained until drive returns |
| Remove folder | Exclusive index removed; original files retained |
| Open twice | Second UI launch explains that another instance is using the library |
| Uninstall retaining data, reinstall | Models and index reused |
| Uninstall removing data | Only application data removed; original image folders retained |
| No GPU or unavailable driver | Actual fallback/errors documented; do not infer B70 performance |

## Audit evidence, 2026-09-09

- Dependency consistency: `pip check` passed after fixing the local SentencePiece metadata ACL.
- Final regression suite: 87 tests passed, including isolated lifecycle and interrupted-download checks.
- Bundled EXE: isolated empty-library `--diagnose` passed using the newly built runtime.
- Windows Sandbox (build 26100, networking and vGPU disabled): two complete install / offline diagnosis / silent uninstall cycles passed. Reinstallation retained the application-data sentinel; an original-file sentinel outside application data survived both cycles. This validates installation on a fresh Windows 11 environment without the host development runtime. Interactive dialogs, the data-removal choice, and standard-user privileges were not exercised by this silent test.
- Model sources: anonymous HEAD requests for all 75 pinned files succeeded. Large-weight cold downloads were not repeated.
- Real GPU test: two generated 1024-square images indexed in an isolated SQLite/USearch library. All stages ready, eight region vectors, OCR and semantic recall nonempty. SigLIP2, PP-OCRv6 and YOLOE-M reported GPU.0.
- Real Qwen INT8: OpenVINO GenAI on GPU.0 returned valid JSON for a blue image (`match=100`, `hard_fail=false`). This is a smoke test, not a relevance or throughput benchmark.
- Regressions cover missing verification markers, selective re-download, hash failure preserving the original, Range resume, Unicode folders, restart, modification, deletion, folder removal and queued-job shutdown.
- Windows 10, a standard-user account, interactive uninstall with data removal, real network-interruption recovery, CPU-only model inference, INT4 and local VLM imports on fresh machines, long-duration GPU load, and dataset redistribution rights remain release acceptance items. Dependency resolution in a freshly created development venv also remains untested; the packaged runtime was tested independently. Do not mark these passed based on unit tests.

## Publishing

Set the real `VISUAL_SEARCH_GITHUB_REPOSITORY` before the production build. Test generated `dist/install.ps1` with the intended release tag. The source installer script contains placeholders by design; it is not a ready-to-run download link.

Publish installer, checksum, SBOM and notices as release assets. Model weights belong on Hugging Face. Keep local assistant settings, environments, logs, databases, caches, credentials and private keys out of Git. Confirm rights to redistribute benchmark images before publishing them. The filename/token scan is not a comprehensive vulnerability audit.
