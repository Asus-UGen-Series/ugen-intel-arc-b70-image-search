from __future__ import annotations

import itertools
import queue
import threading
from contextlib import contextmanager
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class GpuPriority(IntEnum):
    QUERY_EMBEDDING = 0
    VLM_REVIEW = 1
    BACKGROUND_INDEX = 2
    WARMUP = 3


class InferenceCancelled(RuntimeError):
    pass


class GpuRuntimeUnavailable(RuntimeError):
    """The OpenVINO GPU plugin reported a fatal driver/resource failure."""


GPU_FATAL_MARKERS = (
    "CL_OUT_OF_RESOURCES",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
    "DEVICE_LOST",
    "COULD NOT CREATE A MEMORY",
)


def is_gpu_fatal_error(error: BaseException) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in GPU_FATAL_MARKERS)


@dataclass(order=True)
class _WorkItem(Generic[T]):
    priority: int
    sequence: int
    future: Future[T] = field(compare=False)
    function: Callable[[], T] = field(compare=False)
    generation: int | None = field(compare=False, default=None)


class GpuScheduler:
    """One cooperative priority queue for every OpenVINO GPU inference call."""

    def __init__(self, name: str = "VisualSearch-GPU"):
        self._queue: queue.PriorityQueue[_WorkItem] = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._generation = 0
        self._lock = threading.Lock()
        self._interaction_changed = threading.Condition(self._lock)
        self._interactive_count = 0
        self._stopping = threading.Event()
        self._active: _WorkItem | None = None
        self._fault_message = ""
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def next_generation(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def submit(self, function: Callable[[], T], priority: GpuPriority,
               generation: int | None = None) -> Future[T]:
        future: Future[T] = Future()
        with self._lock:
            fault_message = self._fault_message
        if fault_message:
            future.set_exception(GpuRuntimeUnavailable(fault_message))
            return future
        self._queue.put(_WorkItem(int(priority), next(self._sequence), future, function, generation))
        return future

    @contextmanager
    def interactive(self):
        """Prevent new background batches for the complete interactive query."""
        with self._interaction_changed:
            self._interactive_count += 1
        try:
            yield
        finally:
            with self._interaction_changed:
                self._interactive_count = max(0, self._interactive_count - 1)
                self._interaction_changed.notify_all()

    def wait_for_background_slot(self) -> None:
        with self._interaction_changed:
            self._interaction_changed.wait_for(
                lambda: self._interactive_count == 0 or self._stopping.is_set()
            )

    def run(self, function: Callable[[], T], priority: GpuPriority,
            generation: int | None = None, timeout: float | None = None) -> T:
        return self.submit(function, priority, generation).result(timeout=timeout)

    def is_current(self, generation: int | None) -> bool:
        return generation is None or generation == self.generation

    def snapshot(self) -> dict[str, object]:
        active = self._active
        return {
            "queued": self._queue.qsize(),
            "generation": self.generation,
            "active_priority": active.priority if active else None,
            "active_generation": active.generation if active else None,
            "interactive_queries": self._interactive_count,
            "faulted": bool(self._fault_message),
            "fault": self._fault_message,
        }

    def shutdown(self, wait: bool = True) -> None:
        self._stopping.set()
        with self._interaction_changed:
            self._interaction_changed.notify_all()
        self._queue.put(_WorkItem(99, next(self._sequence), Future(), lambda: None))
        if wait:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stopping.is_set():
            item = self._queue.get()
            if self._stopping.is_set():
                break
            if item.future.cancelled():
                continue
            with self._lock:
                fault_message = self._fault_message
            if fault_message:
                item.future.set_exception(GpuRuntimeUnavailable(fault_message))
                continue
            if not self.is_current(item.generation):
                item.future.set_exception(InferenceCancelled("Superseded by a newer search"))
                continue
            self._active = item
            try:
                result = item.function()
                if not self.is_current(item.generation):
                    raise InferenceCancelled("Superseded by a newer search")
                item.future.set_result(result)
            except BaseException as exc:
                if is_gpu_fatal_error(exc):
                    message = (
                        "Intel GPU runtime 發生致命資源錯誤；已停止派發後續推論，"
                        "避免把剩餘圖片全部標成失敗。請重新啟動 Visual Search 後續跑。\n"
                        + str(exc)
                    )
                    with self._lock:
                        self._fault_message = message
                    item.future.set_exception(GpuRuntimeUnavailable(message))
                else:
                    item.future.set_exception(exc)
            finally:
                self._active = None
