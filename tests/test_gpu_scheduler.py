import threading

import pytest

from visual_search.gpu_scheduler import (
    GpuPriority, GpuRuntimeUnavailable, GpuScheduler, InferenceCancelled,
)


def test_gpu_scheduler_runs_interactive_work_before_queued_background_work():
    scheduler = GpuScheduler("test-gpu")
    started = threading.Event()
    release = threading.Event()
    order = []

    def active():
        started.set()
        release.wait(2)

    first = scheduler.submit(active, GpuPriority.BACKGROUND_INDEX)
    assert started.wait(1)
    low = scheduler.submit(lambda: order.append("warmup"), GpuPriority.WARMUP)
    high = scheduler.submit(lambda: order.append("query"), GpuPriority.QUERY_EMBEDDING)
    release.set()
    first.result(2)
    high.result(2)
    low.result(2)
    scheduler.shutdown()
    assert order == ["query", "warmup"]


def test_gpu_scheduler_rejects_superseded_generation():
    scheduler = GpuScheduler("test-generation")
    generation = scheduler.next_generation()
    scheduler.next_generation()
    future = scheduler.submit(lambda: 42, GpuPriority.VLM_REVIEW, generation=generation)
    with pytest.raises(InferenceCancelled):
        future.result(2)
    scheduler.shutdown()


def test_gpu_scheduler_circuit_breaks_after_fatal_opencl_error():
    scheduler = GpuScheduler("test-gpu-fault")

    def fail():
        raise RuntimeError("[GPU] CL_OUT_OF_RESOURCES exception")

    with pytest.raises(GpuRuntimeUnavailable):
        scheduler.run(fail, GpuPriority.BACKGROUND_INDEX, timeout=2)
    with pytest.raises(GpuRuntimeUnavailable):
        scheduler.run(lambda: 42, GpuPriority.QUERY_EMBEDDING, timeout=2)
    assert scheduler.snapshot()["faulted"] is True
    scheduler.shutdown()
