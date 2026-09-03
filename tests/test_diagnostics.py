import json

from visual_search import app as application
from visual_search.paths import AppPaths


def test_safe_diagnostics_do_not_compile_models(tmp_path, monkeypatch):
    paths = AppPaths(
        tmp_path, tmp_path / "metadata.db", tmp_path / "vectors.usearch",
        tmp_path / "thumbnails", tmp_path / "models", tmp_path / "logs",
        tmp_path / "config.json",
    )
    paths.ensure()
    monkeypatch.setattr(
        application, "detect_openvino_devices",
        lambda: {"available": ["CPU", "GPU.0"], "selected": "GPU.0", "ok": True},
    )
    monkeypatch.setattr(
        application, "create_services",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("diagnose compiled models")),
    )

    assert application.diagnose(paths) == 0
    report = json.loads((paths.logs / "latest-diagnostics.json").read_text(encoding="utf-8"))
    assert report["diagnostic_mode"] == "safe-no-model-compile"
    assert report["gpu_scheduler"]["running"] is False
    assert all(row["loaded"] is False for row in report["model_execution"])
