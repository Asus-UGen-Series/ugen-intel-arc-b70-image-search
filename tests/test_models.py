import hashlib
import json

import numpy as np

from visual_search.engines import resize_position_embeddings
from visual_search.models import ModelManager, ModelRegistry


def test_position_resize_keeps_static_patch_shape():
    base = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    result = resize_position_embeddings(base, 8, 16, 256)
    assert result.shape == (1, 256, 3)
    assert np.isfinite(result).all()


def test_position_resize_rejects_more_than_max_patches():
    base = np.zeros((16, 16, 8), dtype=np.float32)
    try:
        resize_position_embeddings(base, 17, 16, 256)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid grid to be rejected")


def test_model_manager_requires_immutable_manifest(tmp_path):
    payload = b"openvino"
    model_file = tmp_path / "model" / "vision_model.xml"
    model_file.parent.mkdir()
    model_file.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"model": {
        "id": "test", "source_repo": "owner/repo", "revision": "abc123",
        "files": [{"name": "vision_model.xml", "url": "https://example.invalid/file", "sha256": digest, "size": len(payload)}],
    }}), encoding="utf-8")
    manager = ModelManager(model_file.parent, manifest)
    assert manager.configured()
    assert manager.downloaded()
    assert not manager.ready()
    manager.mark_verified("CPU")
    assert manager.ready()


def test_model_manager_can_select_ocr_manifest_section(tmp_path):
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"model": {}, "ocr_model": {
        "id": "ocr-test", "source_repo": "owner/ocr", "revision": "def456", "files": [],
    }}), encoding="utf-8")
    manager = ModelManager(tmp_path / "ocr", manifest, "ocr_model")
    assert manager.model_id == "ocr-test"
    assert manager.source_repo == "owner/ocr"


def test_model_manager_accepts_a_remote_bundle_path(tmp_path):
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"model": {
        "id": "model-test", "source_repo": "owner/models", "revision": "a" * 40,
        "files": [{
            "name": "vision_model.xml",
            "source_name": "siglip/vision_model.xml",
            "url": "https://huggingface.co/owner/models/resolve/" + "a" * 40 + "/siglip/vision_model.xml",
            "sha256": "b" * 64,
            "size": 10,
        }],
    }}), encoding="utf-8")

    item = ModelManager(tmp_path / "model", manifest).files()[0]

    assert item.name == "vision_model.xml"
    assert item.source_name == "siglip/vision_model.xml"


def test_model_registry_rejects_zip_path_traversal(tmp_path):
    import zipfile

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "no")
    manifest = tmp_path / "models.json"
    manifest.write_text('{"manifest_version":4}', encoding="utf-8")
    registry = ModelRegistry(tmp_path / "models", manifest)
    try:
        registry.import_local_vlm(archive)
    except ValueError as exc:
        assert "Unsafe ZIP" in str(exc)
    else:
        raise AssertionError("unsafe archive was accepted")


def test_model_registry_imports_valid_prompt_free_object_package(tmp_path):
    source = tmp_path / "converted-yoloe"
    source.mkdir()
    (source / "model.xml").write_text("<net/>", encoding="utf-8")
    (source / "model.bin").write_bytes(b"weights")
    (source / "metadata.yaml").write_text(
        "visual_search_family: yoloe26_pf\nprompt_mode: prompt-free\nscale: S\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"yoloe_s": {"directory": "yoloe-s"}}), encoding="utf-8")
    registry = ModelRegistry(tmp_path / "models", manifest)

    target = registry.import_local_object(source, "yoloe26s-pf")

    assert target == tmp_path / "models" / "yoloe-s"
    assert (target / "model.xml").is_file()
    assert registry.validate_object_directory(target, "yoloe26s-pf")["scale"] == "S"


def test_model_registry_rejects_object_scale_mismatch(tmp_path):
    source = tmp_path / "wrong-scale"
    source.mkdir()
    (source / "model.xml").write_text("<net/>", encoding="utf-8")
    (source / "model.bin").write_bytes(b"weights")
    (source / "metadata.yaml").write_text("prompt_mode: prompt-free\nscale: M\n", encoding="utf-8")
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"yoloe_s": {"directory": "yoloe-s"}}), encoding="utf-8")
    registry = ModelRegistry(tmp_path / "models", manifest)

    try:
        registry.import_local_object(source, "yoloe26s-pf")
    except ValueError as exc:
        assert "標示為 M" in str(exc)
    else:
        raise AssertionError("mismatched model scale was accepted")
