from pathlib import Path
import time

from visual_search.library_monitor import LibraryMonitor


class Repository:
    def get_by_path(self, _path):
        return None


class Controller:
    def __init__(self):
        self.repository = Repository()
        self.scanned = []
        self.removed = []

    def scan(self, root, progress=None, selected_paths=None, cleanup=True):
        self.scanned.append((root, selected_paths, cleanup))

    def remove_path(self, path):
        self.removed.append(path)
        return True


def test_monitor_indexes_only_stable_changed_file(tmp_path):
    controller = Controller()
    image = tmp_path / "new.png"
    image.write_bytes(b"image")
    monitor = LibraryMonitor(controller, [str(tmp_path)], stable_seconds=0.01)

    monitor._process("upsert", image, None)

    assert controller.scanned == [(tmp_path, [image], False)]


def test_monitor_move_outside_root_removes_old_asset(tmp_path):
    controller = Controller()
    source = tmp_path / "old.png"
    destination = tmp_path.parent / "outside.png"
    monitor = LibraryMonitor(controller, [str(tmp_path)], stable_seconds=0.01)

    monitor._process("move", source, destination)

    assert controller.removed == [source]


def test_watchdog_dispatches_real_create_event(tmp_path):
    controller = Controller()
    monitor = LibraryMonitor(
        controller, [str(tmp_path)], debounce_seconds=0.05,
        stable_seconds=0.05, reconcile_seconds=60,
    )
    monitor.start()
    try:
        image = tmp_path / "event.png"
        image.write_bytes(b"event")
        deadline = time.monotonic() + 3
        while not controller.scanned and time.monotonic() < deadline:
            time.sleep(0.05)
        assert controller.scanned
        assert controller.scanned[-1] == (tmp_path, [image], False)
        assert monitor.snapshot()["watching"] is True
    finally:
        monitor.stop()


def test_monitor_can_remove_root_without_touching_files(tmp_path):
    controller = Controller()
    image = tmp_path / "kept.png"
    image.write_bytes(b"image")
    monitor = LibraryMonitor(controller, [str(tmp_path)], stable_seconds=0.01)

    assert monitor.remove_root(tmp_path) is True
    assert monitor.snapshot()["roots"] == 0
    assert monitor.snapshot()["watching"] is False
    assert image.exists()
