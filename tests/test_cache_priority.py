# SPDX-License-Identifier: Apache-2.0
"""Request cache priority, persistent promotion, and bounded eviction."""

from pathlib import Path

import pytest

from omlx.cache.paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheManager,
)


def block(n, priority="normal", size=10):
    return PagedSSDBlockMetadata(
        block_hash=n.to_bytes(32, "big"),
        file_path=Path(f"{n}.safetensors"),
        file_size=size,
        token_count=4,
        created_at=float(n),
        last_access=float(n),
        num_layers=1,
        cache_priority=priority,
    )


def test_protected_blocks_are_skipped_and_walk_terminates(tmp_path):
    manager = PagedSSDCacheManager(
        tmp_path, 20, referenced_hashes=lambda: {block(1).block_hash}
    )
    try:
        manager._index.add(block(1, "low"))
        manager._index.add(block(2, "low"))
        evicted = manager._evict_tracked_until_size(0)
        assert [b.block_hash for _, b in evicted] == [block(2).block_hash]
        assert manager._evict_tracked_until_size(0) == []
    finally:
        manager.close()


def test_legacy_metadata_and_round_trip():
    metadata = block(1, "low")
    assert PagedSSDBlockMetadata.from_dict(metadata.to_dict()).cache_priority == "low"
    legacy = metadata.to_dict()
    legacy.pop("cache_priority")
    assert PagedSSDBlockMetadata.from_dict(legacy).cache_priority == "normal"


def test_default_priority_preserves_multi_index_lru_trace(tmp_path):
    from omlx.cache.paged_ssd_cache import GDNCheckpointMetadata

    manager = PagedSSDCacheManager(tmp_path, 50)
    try:
        # Insertion order within an index takes precedence over timestamps,
        # matching get_lru_entries(1).
        for n in [3, 1, 5]:
            manager._index.add(block(n))
        manager._incompatible_index.add(block(2))
        sidecar = GDNCheckpointMetadata(
            block(4).block_hash,
            "0" * 64,
            tmp_path / "sidecar",
            10,
            4,
            "test",
            4,
            4.0,
            4.0,
        )
        manager._gdn_sidecar_index.add(sidecar)
        evicted = manager._evict_tracked_until_size(0)
        assert [
            int.from_bytes(
                (
                    b.source_block_hash
                    if hasattr(b, "source_block_hash")
                    else b.block_hash
                ),
                "big",
            )
            for _, b in evicted
        ] == [2, 3, 1, 4, 5]
    finally:
        manager.close()


def test_promotion_updates_low_byte_accounting(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 20)
    try:
        metadata = block(1, "low")
        metadata.file_path = tmp_path / "block.safetensors"
        manager._index.add(metadata)
        manager.promote_block(metadata.block_hash)
        assert metadata.cache_priority == "normal"
        assert manager.low_priority_size == 0
        manager._index.update_file_size(metadata.block_hash, 30)
        assert manager.low_priority_size == 0
    finally:
        manager.close()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("low", {"cache_priority": "low"}),
        (" LOW ", {"cache_priority": "low"}),
        ("normal", {}),
        ("invalid", {}),
    ],
)
def test_header(value, expected):
    from starlette.requests import Request

    from omlx.server import _cache_store_kwargs

    request = Request(
        {"type": "http", "headers": [(b"x-omlx-cache-priority", value.encode())]}
    )
    assert _cache_store_kwargs(request) == expected


@pytest.fixture
def ssd(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 100_000, expected_block_size_tokens=4)
    yield manager
    manager.close()


def save(manager, n, priority="normal"):
    import mlx.core as mx

    data = [(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))]
    result = manager.save_block(block(n).block_hash, data, 4, cache_priority=priority)
    assert manager.wait_for_pending_writes(timeout=5.0)
    return result


def test_priority_survives_restart_and_promotion_rebuild(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 100_000, expected_block_size_tokens=4)
    assert save(manager, 1, "low")
    assert save(manager, 2, "low")
    manager.close()
    manager = PagedSSDCacheManager(tmp_path, 100_000, expected_block_size_tokens=4)
    assert manager.get_block_metadata(block(1).block_hash).cache_priority == "low"
    assert save(manager, 1, "low")
    assert manager.get_block_metadata(block(1).block_hash).cache_priority == "low"
    manager.promote_block(block(1).block_hash)
    assert manager.get_block_metadata(block(1).block_hash).cache_priority == "normal"
    assert save(manager, 1, "low")  # Never demote a shared interactive prefix.
    manager.close()
    manager = PagedSSDCacheManager(tmp_path, 100_000, expected_block_size_tokens=4)
    try:
        assert (
            manager.get_block_metadata(block(1).block_hash).cache_priority == "normal"
        )
        assert manager.get_block_metadata(block(2).block_hash).cache_priority == "low"
    finally:
        manager.close()


def test_pending_writes_are_never_eviction_candidates(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 20)
    try:
        manager._index.add(block(1, "low"))
        manager._index.add(block(2))
        with manager._pending_write_hashes_lock:
            manager._pending_write_hashes.add(block(1).block_hash)
        evicted = manager._evict_tracked_until_size(0)
        assert [b.block_hash for _, b in evicted] == [block(2).block_hash]
    finally:
        manager._clear_pending_write(block(1).block_hash)
        manager.close()


def test_prefix_store_protects_its_chain_and_keeps_priority(tmp_path):
    from types import SimpleNamespace

    import mlx.core as mx

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(
        block_size=4, max_blocks=16, model_name="test", initial_blocks=16
    )

    def referenced():
        return {
            b.block_hash for b in paged.allocated_blocks.values() if b.ref_count > 0
        }

    manager = PagedSSDCacheManager(
        tmp_path,
        100_000,
        retained_request_count=2,
        expected_block_size_tokens=4,
        referenced_hashes=referenced,
    )
    paged.set_paged_ssd_cache_manager(manager)
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(layers=[object()]),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=manager,
    )
    tensor = mx.zeros((1, 1, 12, 2))
    try:
        table = prefix.store_cache(
            "batch",
            list(range(12)),
            [{"state": (tensor, tensor), "cache_type": "KVCache", "meta_state": ()}],
            cache_priority="low",
        )
        assert manager.wait_for_pending_writes(5.0)
        assert table is not None
        metadata = manager._index.get_all_metadata()
        assert len(metadata) == 3  # Never evict this request's earlier blocks.
        assert all(m.cache_priority == "low" for m in metadata)
        assert all(
            manager.has_block(paged.allocated_blocks[bid].block_hash)
            for bid in table.block_ids
        )
    finally:
        manager.close()


def test_sidecar_inherits_priority_and_promotion_survives_rebuild(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, 100_000, expected_block_size_tokens=4)
    try:
        assert save(manager, 1, "low")
        meta = manager.get_block_metadata(block(1).block_hash)
        staged = tmp_path / "stage.safetensors"
        staged.write_bytes(b"synthetic recurrent state")
        path = manager.commit_gdn_checkpoint_file(
            meta.block_hash,
            staged,
            token_count=4,
            model_name="test",
            cache_signature=meta.cache_signature,
            block_size=4,
        )
        assert path is not None
        sidecar = manager._gdn_sidecar_index.get_all_metadata()[0]
        assert sidecar.cache_priority == "low"
        assert manager.low_priority_size == meta.file_size + sidecar.file_size
        manager.promote_block(meta.block_hash)
        assert manager.low_priority_size == 0
    finally:
        manager.close()
    manager = PagedSSDCacheManager(tmp_path, 100_000)
    try:
        assert (
            manager._gdn_sidecar_index.get_all_metadata()[0].cache_priority == "normal"
        )
        assert (
            manager.get_block_metadata(block(1).block_hash).cache_priority == "normal"
        )
    finally:
        manager.close()


def test_scheduler_initializes_live_manager_with_budget_and_references(
    tmp_path, monkeypatch
):
    import omlx.scheduler as module
    from omlx.cache.paged_cache import PagedCacheManager

    scheduler = module.Scheduler.__new__(module.Scheduler)
    scheduler.config = module.SchedulerConfig(
        paged_ssd_cache_dir=str(tmp_path), max_num_seqs=3
    )
    scheduler.memory_monitor = None
    scheduler.block_aware_cache = None
    scheduler.paged_cache_manager = PagedCacheManager(
        block_size=4, max_blocks=16, initial_blocks=16
    )
    monkeypatch.setattr(module, "BoundarySnapshotSSDStore", None)
    assert scheduler._init_tiered_cache()
    try:
        assert scheduler.paged_ssd_cache_manager._retained_request_count == 3
        b = scheduler.paged_cache_manager.allocate_block()
        b.block_hash = block(1).block_hash
        assert block(
            1
        ).block_hash in scheduler.paged_ssd_cache_manager._protected_hashes(
            incoming_low=True
        )
    finally:
        scheduler.paged_ssd_cache_manager.close()


def test_real_save_fails_when_protected_bytes_fill_cache(ssd):
    assert save(ssd, 1, "low")
    original = ssd.get_block_metadata(block(1).block_hash)
    ssd._max_size = original.file_size
    ssd._referenced_hashes = lambda: {original.block_hash}
    assert not save(ssd, 2, "low")
    assert ssd._index.total_size <= ssd._max_size
    assert ssd.get_block_metadata(block(1).block_hash) is original
    assert ssd.get_block_metadata(block(2).block_hash) is None


def test_promotion_io_failure_does_not_fail_inference(ssd, monkeypatch, caplog):
    assert save(ssd, 1, "low")

    def fail(*args):
        raise OSError("synthetic full disk")

    monkeypatch.setattr(ssd, "_write_priority_marker", fail)
    ssd.promote_block(block(1).block_hash)
    assert ssd.get_block_metadata(block(1).block_hash).cache_priority == "normal"
    assert "Cannot persist cache-priority promotion" in caplog.text
