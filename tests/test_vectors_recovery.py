from __future__ import annotations

from pathlib import Path

import numpy as np

import visual_search.vectors as vector_module
from visual_search.vectors import USearchVectorIndex, validate_index_file


def test_atomic_save_and_isolated_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "vectors.usearch"
    original = USearchVectorIndex(path, 4)
    original.add(7, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    original.save()

    assert validate_index_file(path, 4, "f32") is True
    restored = USearchVectorIndex(path, 4)
    assert restored.size() == 1
    assert restored.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 1)[0][0] == 7
    assert list(tmp_path.glob("*.tmp")) == []


def test_unsafe_index_is_quarantined_for_sqlite_rebuild(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "vectors.usearch"
    path.write_bytes(b"not-a-usearch-index")
    monkeypatch.setattr(vector_module, "validate_index_file", lambda *_args, **_kwargs: False)

    recovered = USearchVectorIndex(path, 4)

    assert recovered.size() == 0
    assert recovered.recovered_from is not None
    assert recovered.recovered_from.read_bytes() == b"not-a-usearch-index"
    assert not path.exists()
