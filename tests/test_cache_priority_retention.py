# SPDX-License-Identifier: Apache-2.0
"""Bounded completion history, shared-prefix retention, and eviction tiers."""

from pathlib import Path

import pytest

from omlx.cache.paged_ssd_cache import PagedSSDBlockMetadata, PagedSSDCacheManager


def metadata(n, priority="low", size=10):
    return PagedSSDBlockMetadata(
        n.to_bytes(32, "big"),
        Path(f"{n}.safetensors"),
        size,
        4,
        float(n),
        float(n),
        1,
        cache_priority=priority,
    )


@pytest.mark.parametrize("incoming", ["normal", "low"])
def test_any_save_evicts_unretained_low_then_normal_then_retained(tmp_path, incoming):
    manager = PagedSSDCacheManager(tmp_path, 40, retained_request_count=2)
    try:
        for n, priority in [(1, "normal"), (2, "low"), (3, "low"), (4, "normal")]:
            manager._index.add(metadata(n, priority))
        manager.record_low_priority_request("recent", [metadata(3).block_hash])
        assert manager._admit_priority_block(metadata(5, incoming))
        assert manager._index.get(metadata(2).block_hash) is None
        evicted = manager._evict_tracked_until_size(0, incoming_low=incoming == "low")
        assert [int.from_bytes(m.block_hash, "big") for _, m in evicted] == (
            [1, 4, 3] if incoming == "low" else [3, 1, 4]
        )
    finally:
        manager._clear_pending_write(metadata(5).block_hash)
        manager.close()


def test_last_m_completed_chains_use_hash_union_with_shared_references(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 40, retained_request_count=2)
    try:
        for n in range(1, 5):
            manager._index.add(metadata(n))
        manager.record_low_priority_request(
            "one", [metadata(1).block_hash, metadata(2).block_hash]
        )
        manager.record_low_priority_request(
            "two", [metadata(2).block_hash, metadata(3).block_hash]
        )
        manager.record_low_priority_request(
            "three", [metadata(3).block_hash, metadata(4).block_hash]
        )
        assert manager.retained_low_priority_size == 30
        evicted = manager._evict_tracked_until_size(30)
        assert [m.block_hash for _, m in evicted] == [metadata(1).block_hash]
        # Repeating a completion is idempotent and does not occupy two slots.
        manager.record_low_priority_request(
            "three", [metadata(3).block_hash, metadata(4).block_hash]
        )
        assert manager.retained_low_priority_size == 30
    finally:
        manager.close()


def test_completion_history_survives_restart(tmp_path):
    import mlx.core as mx

    manager = PagedSSDCacheManager(tmp_path, 100_000, retained_request_count=2)
    tensor = mx.zeros((1, 1, 4, 2))
    for n in range(1, 4):
        assert manager.save_block(
            metadata(n).block_hash, [(tensor, tensor)], 4, cache_priority="low"
        )
    assert manager.wait_for_pending_writes(5.0)
    manager.record_low_priority_request("older", [metadata(1).block_hash])
    manager.record_low_priority_request("latest", [metadata(3).block_hash])
    manager.close()
    manager = PagedSSDCacheManager(tmp_path, 100_000, retained_request_count=1)
    try:
        assert (
            manager.retained_low_priority_size
            == manager.get_block_metadata(metadata(3).block_hash).file_size
        )
        evicted = manager._evict_tracked_until_size(0, incoming_low=True)
        assert evicted[-1][1].block_hash == metadata(3).block_hash
    finally:
        manager.close()


def test_promotion_removes_retained_bytes(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 100, retained_request_count=1)
    try:
        m = metadata(1)
        m.file_path = tmp_path / "one.safetensors"
        manager._index.add(m)
        manager.record_low_priority_request("one", [m.block_hash])
        assert manager.retained_low_priority_size == 10
        manager.promote_block(m.block_hash)
        assert manager.retained_low_priority_size == 0
        assert manager.low_priority_size == 0
    finally:
        manager.close()


@pytest.mark.parametrize("priority,expected", [("normal", "normal"), ("low", "low")])
def test_scheduler_promotes_only_normal_reuse(tmp_path, priority, expected):
    from types import SimpleNamespace

    from omlx.scheduler import Scheduler

    manager = PagedSSDCacheManager(tmp_path, 100)
    m = metadata(1)
    m.file_path = tmp_path / "one.safetensors"
    manager._index.add(m)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.paged_ssd_cache_manager = manager
    scheduler.paged_cache_manager = SimpleNamespace(
        allocated_blocks={7: SimpleNamespace(block_hash=m.block_hash)}
    )
    request = SimpleNamespace(
        cache_priority=priority,
        cached_tokens=4,
        block_table=SimpleNamespace(block_ids=[7]),
    )
    try:
        scheduler._promote_prefix_cache(request)
        assert m.cache_priority == expected
    finally:
        manager.close()


def test_completion_records_hashes_before_physical_ids_are_reused(tmp_path):
    import threading
    from types import SimpleNamespace

    from omlx.request import Request, RequestStatus, SamplingParams
    from omlx.scheduler import Scheduler

    manager = PagedSSDCacheManager(tmp_path, 100, retained_request_count=1)
    scheduler = Scheduler.__new__(Scheduler)
    request = Request(
        request_id="done",
        prompt=[1],
        sampling_params=SamplingParams(),
        cache_priority="low",
    )
    request.status = RequestStatus.FINISHED_STOPPED
    scheduler.requests = {"done": request}
    block = SimpleNamespace(block_hash=metadata(1).block_hash)
    scheduler.paged_cache_manager = SimpleNamespace(
        _lock=threading.RLock(), allocated_blocks={7: block}
    )
    scheduler.paged_ssd_cache_manager = manager
    try:
        scheduler._record_low_priority_request("done", SimpleNamespace(block_ids=[7]))
        block.block_hash = metadata(2).block_hash
        assert manager._eviction_priority(metadata(1), incoming_low=True) == 2
        assert manager._eviction_priority(metadata(2)) == 0
    finally:
        manager.close()


def test_reference_acquisition_is_atomic_with_eviction(tmp_path):
    import threading

    lock = threading.RLock()
    referenced = set()
    started, done = threading.Event(), threading.Event()
    result = []
    manager = PagedSSDCacheManager(
        tmp_path, 20, reference_lock=lock, referenced_hashes=lambda: set(referenced)
    )
    manager._index.add(metadata(1))
    manager._index.add(metadata(2))

    def evict():
        started.set()
        result.extend(manager._evict_tracked_until_size(0))
        done.set()

    try:
        with lock:
            thread = threading.Thread(target=evict)
            thread.start()
            assert started.wait(2.0)
            referenced.add(metadata(1).block_hash)
        assert done.wait(2.0)
        thread.join()
        assert [m.block_hash for _, m in result] == [metadata(2).block_hash]
    finally:
        manager.close()


@pytest.mark.parametrize("incoming", ["low", "normal"])
def test_exhausted_cache_skips_store_without_eviction_or_size_growth(
    tmp_path, incoming
):
    import threading

    manager = PagedSSDCacheManager(
        tmp_path,
        20,
        retained_request_count=1,
        referenced_hashes=lambda: {metadata(1).block_hash},
        reference_lock=threading.RLock(),
    )
    try:
        manager._index.add(metadata(1, size=15))
        manager._index.add(metadata(2, "normal", size=5))
        assert not manager._admit_priority_block(metadata(3, incoming, size=10))
        assert manager._index.total_size == 20
        assert manager._index.get(metadata(1).block_hash) is not None
        assert manager._index.get(metadata(2).block_hash) is not None
        assert not manager._pending_write_hashes
    finally:
        manager.close()


def test_no_header_keeps_main_pending_eviction_behavior(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 20)
    try:
        manager._index.add(metadata(1, "normal"))
        manager._index.add(metadata(2, "normal"))
        manager._pending_write_hashes.add(metadata(1).block_hash)
        evicted = manager._evict_tracked_until_size(10)
        assert [m.block_hash for _, m in evicted] == [metadata(1).block_hash]
    finally:
        manager._clear_pending_write(metadata(1).block_hash)
        manager.close()
