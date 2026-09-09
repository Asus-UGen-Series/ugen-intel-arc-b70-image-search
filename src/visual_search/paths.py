from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    database: Path
    vectors: Path
    thumbnails: Path
    models: Path
    logs: Path
    config: Path

    @classmethod
    def default(cls) -> "AppPaths":
        override = os.environ.get("VISUAL_SEARCH_DATA_DIR")
        base = Path(override).resolve() if override else Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "VisualSearch"
        return cls(base, base / "metadata.db", base / "vectors.usearch", base / "thumbnails", base / "models", base / "logs", base / "config.json")

    def ensure(self) -> None:
        for folder in (self.root, self.thumbnails, self.models, self.logs):
            folder.mkdir(parents=True, exist_ok=True)
