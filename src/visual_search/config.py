from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    setup_complete: bool = False
    roots: list[str] = field(default_factory=list)
    device: str = "AUTO"
    ocr_enabled: bool = True
    active_object_profile: str = "yoloe26s-pf"
    vlm_policy: str = "auto"
    vlm_setup_complete: bool = False
    active_vlm_profile: str = "qwen35-9b-int8"
    vlm_local_path: str = ""
    vlm_candidate_count: int = 30
    vlm_timeout_seconds: int = 60
    vlm_image_size: int = 448

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            allowed = set(cls.__dataclass_fields__)
            config = cls(**{key: value for key, value in payload.items() if key in allowed})
            if config.vlm_policy not in {"auto", "always", "off"}:
                config.vlm_policy = "auto"
            if config.active_object_profile not in {"yoloe26s-pf", "yoloe26m-pf"}:
                config.active_object_profile = "yoloe26s-pf"
            config.vlm_candidate_count = max(1, min(int(config.vlm_candidate_count), 30))
            config.vlm_timeout_seconds = max(5, min(int(config.vlm_timeout_seconds), 300))
            if config.vlm_image_size not in {336, 384, 448}:
                config.vlm_image_size = 448
            return config
        except Exception:
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
