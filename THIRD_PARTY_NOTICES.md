# Third-party notices

Visual Search is distributed under AGPL-3.0-only because it integrates the
Ultralytics YOLOE-26 Prompt-Free model. Public releases must make the complete corresponding
application source, build scripts and configuration available.

- Qwen3.5 and its OpenVINO conversions: Apache-2.0.
- OpenVINO, OpenVINO GenAI and Optimum Intel: Apache-2.0.
- PyTorch and TorchVision: BSD-3-Clause-style licenses. PyInstaller-collected
  third-party license texts are retained under `licenses/torch-third-party`
  with flattened, hash-prefixed names to remain compatible with Windows paths.
- SigLIP2 model and Transformers integration: see the pinned model repository.
- PaddleOCR / PP-OCRv6: Apache-2.0.
- Ultralytics YOLOE-26 code and pretrained weights: AGPL-3.0 unless covered by a
  separately purchased Ultralytics Enterprise license.
- PySide6/Qt: LGPL-3.0/GPL/commercial terms as distributed by Qt for Python.
- USearch: Apache-2.0.

Exact installed package versions are recorded in `sbom.json` next to the
application executable.
