from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .indexer import SUPPORTED, IndexController


@dataclass(slots=True)
class LibraryMonitorSnapshot:
    watching: bool = False
    roots: int = 0
    pending: int = 0
    last_scan: str = ""
    last_event: str = ""
    error: str = ""


class LibraryMonitor:
    """Recursive filesystem events plus periodic reconciliation for local libraries."""

    def __init__(self, controller: IndexController, roots: list[str], debounce_seconds: float = 2.0,
                 stable_seconds: float = 1.0, reconcile_seconds: float = 15 * 60):
        self.controller = controller
        self.debounce_seconds = max(0.1, float(debounce_seconds))
        self.stable_seconds = max(0.1, float(stable_seconds))
        self.reconcile_seconds = max(5.0, float(reconcile_seconds))
        self._roots: dict[str, Path] = {}
        for root in roots:
            self._remember_root(Path(root))
        self._events: queue.Queue[tuple[str, Path, Path | None]] = queue.Queue()
        self._pending: dict[str, tuple[float, str, Path, Path | None]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observer = None
        self._worker: threading.Thread | None = None
        self._snapshot = LibraryMonitorSnapshot(roots=len(self._roots))

    @staticmethod
    def _key(path: Path) -> str:
        try:
            return str(path.resolve()).casefold()
        except OSError:
            return str(path.absolute()).casefold()

    def _remember_root(self, root: Path) -> None:
        self._roots[self._key(root)] = root

    def add_root(self, root: Path) -> None:
        key = self._key(root)
        if key in self._roots:
            return
        self._roots[key] = root
        with self._lock:
            self._snapshot.roots = len(self._roots)
        if self._observer is not None and root.is_dir():
            self._observer.schedule(self._handler(), str(root), recursive=True)

    def _handler(self):
        from watchdog.events import FileSystemEventHandler

        monitor = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    monitor.queue("upsert", Path(event.src_path))

            def on_modified(self, event):
                if not event.is_directory:
                    monitor.queue("upsert", Path(event.src_path))

            def on_deleted(self, event):
                if not event.is_directory:
                    monitor.queue("delete", Path(event.src_path))

            def on_moved(self, event):
                if not event.is_directory:
                    monitor.queue("move", Path(event.src_path), Path(event.dest_path))

        return Handler()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            from watchdog.observers import Observer

            self._stop.clear()
            self._observer = Observer()
            for root in self._roots.values():
                if root.is_dir():
                    self._observer.schedule(self._handler(), str(root), recursive=True)
            self._observer.start()
            self._worker = threading.Thread(target=self._run, name="VisualSearch-LibraryMonitor", daemon=True)
            self._worker.start()
            with self._lock:
                self._snapshot.watching = True
                self._snapshot.error = ""
        except Exception as exc:
            with self._lock:
                self._snapshot.watching = False
                self._snapshot.error = str(exc)

    def stop(self) -> None:
        self._stop.set()
        observer = self._observer
        if observer is not None:
            observer.stop()
            observer.join(timeout=5)
        worker = self._worker
        if worker is not None:
            worker.join(timeout=5)
        with self._lock:
            self._snapshot.watching = False

    def queue(self, kind: str, source: Path, destination: Path | None = None) -> None:
        self._events.put((kind, source, destination))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            value = LibraryMonitorSnapshot(**asdict(self._snapshot))
            value.pending = len(self._pending) + self._events.qsize()
            return asdict(value)

    def _root_for(self, path: Path) -> Path | None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        for root in self._roots.values():
            try:
                if resolved.is_relative_to(root.resolve()):
                    return root
            except (OSError, ValueError):
                continue
        return None

    def _stable(self, path: Path) -> bool:
        previous = None
        for _attempt in range(6):
            if self._stop.is_set() or not path.is_file():
                return False
            try:
                stat = path.stat()
                current = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                current = None
            if current is not None and current == previous:
                return True
            previous = current
            self._stop.wait(self.stable_seconds)
        return False

    def _process(self, kind: str, source: Path, destination: Path | None) -> None:
        if kind == "delete":
            self.controller.remove_path(source)
        elif kind == "move":
            target = destination or source
            root = self._root_for(target)
            if root is not None and target.suffix.casefold() in SUPPORTED and self._stable(target):
                self.controller.scan(root, selected_paths=[target], cleanup=False)
                if not source.exists() and self.controller.repository.get_by_path(str(source)) is not None:
                    self.controller.remove_path(source)
            else:
                self.controller.remove_path(source)
        elif source.suffix.casefold() in SUPPORTED:
            root = self._root_for(source)
            if root is not None and self._stable(source):
                self.controller.scan(root, selected_paths=[source], cleanup=False)
        with self._lock:
            self._snapshot.last_event = f"{kind}: {destination or source}"
            self._snapshot.error = ""

    def _reconcile(self) -> None:
        for root in list(self._roots.values()):
            if self._stop.is_set():
                return
            if root.is_dir():
                self.controller.scan(root)
        with self._lock:
            self._snapshot.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._snapshot.error = ""

    def _run(self) -> None:
        next_reconcile = time.monotonic() + self.reconcile_seconds
        while not self._stop.is_set():
            try:
                kind, source, destination = self._events.get(timeout=0.25)
                key = self._key(destination or source)
                self._pending[key] = (
                    time.monotonic() + self.debounce_seconds, kind, source, destination,
                )
                while True:
                    kind, source, destination = self._events.get_nowait()
                    key = self._key(destination or source)
                    self._pending[key] = (
                        time.monotonic() + self.debounce_seconds, kind, source, destination,
                    )
            except queue.Empty:
                pass
            now = time.monotonic()
            due = [key for key, value in self._pending.items() if value[0] <= now]
            for key in due:
                _deadline, kind, source, destination = self._pending.pop(key)
                try:
                    self._process(kind, source, destination)
                except Exception as exc:
                    with self._lock:
                        self._snapshot.error = str(exc)
            if now >= next_reconcile:
                try:
                    self._reconcile()
                except Exception as exc:
                    with self._lock:
                        self._snapshot.error = str(exc)
                next_reconcile = time.monotonic() + self.reconcile_seconds
