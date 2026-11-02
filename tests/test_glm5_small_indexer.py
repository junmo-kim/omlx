# SPDX-License-Identifier: Apache-2.0
"""Small-query GLM indexer scores, masks, selection and capacity strides."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


@pytest.fixture
def indexer():
    apply_mlx_vlm_glm5_next_compat_patch()
    return importlib.import_module("mlx_vlm.models.glm5_next.small_indexer")


def _reference(indexer, q, keys, weights, before, plen, kpool):
    # The single-sequence SIMD kernel remains an independent arithmetic reference.
    return mx.concatenate(
        [
            mx.pad(
                indexer.small_scores(
                    q[b : b + 1],
                    keys[b : b + 1, :n],
                    weights[b : b + 1],
                    before[b],
                    kpool,
                ),
                ((0, 0), (0, 0), (0, keys.shape[1] - n)),
                constant_values=-1e30,
            )
            for b, n in enumerate(plen)
        ]
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("extent", [1, 7, 8, 9, 513])
def test_scores_match_simd_reference(indexer, dtype, width, extent):
    mx.random.seed(extent + width)
    q = mx.random.normal((2, width, 32, 128)).astype(dtype)
    keys = mx.random.normal((2, extent, 128)).astype(dtype)
    weights = (mx.random.normal((2, width, 32)) * 0.01).astype(dtype)
    plen = [extent, max(1, extent - 3)]
    before = [extent * 4 - 2, max(0, plen[1] * 4 - 3)]
    actual = indexer.small_scores_batched(q, keys, weights, before, plen, 4)
    expected = _reference(indexer, q, keys, weights, before, plen, 4)
    valid = expected > -1e29
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert mx.array_equal(actual <= -1e29, ~valid).item()
    error = mx.where(valid, actual.astype(mx.float32) - expected, 0)
    assert mx.max(mx.abs(error)).item() <= 0.0078125


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("kpool", [1, 4])
def test_growing_pool_preserves_scores_and_topk(indexer, dtype, width, kpool):
    # Binary fractions give exact, tied scores so selection order is testable
    # without assuming that reordered floating-point sums always rank alike.
    mx.random.seed(8)
    q = mx.ones((2, width, 32, 128), dtype=dtype)
    weights = mx.full((2, width, 32), 1 / 32, dtype=dtype)
    capacity = 544
    backing = mx.broadcast_to(
        (mx.random.randint(-32, 33, (2, capacity, 1)) / 128).astype(dtype),
        (2, capacity, 128),
    )
    # Materialize a contiguous capacity buffer, just like PoolingCache.
    backing = mx.contiguous(backing)
    mx.eval(backing)
    for extent in [511, 512, 513, 519]:
        keys = backing[:, :extent]
        cache = SimpleNamespace(_pool_buf=backing, _pool_extent=extent)
        plen = [extent, extent - 7]
        before = [extent * kpool - 2, (extent - 8) * kpool - 1]
        config = SimpleNamespace(index_topk=512 * kpool, index_kpool=kpool)

        def select(scores, count):
            return mx.argsort(-scores, axis=-1)[..., :count]

        selected, actual = indexer.batched_windows(
            config, q, keys, weights, before, plen, select, cache
        )
        expected = _reference(indexer, q, keys, weights, before, plen, kpool)
        assert mx.array_equal(actual, expected).item()
        assert mx.array_equal(selected, select(expected, min(512, extent))).item()


@pytest.mark.parametrize("shape", [(2, 3, 32, 128), (2, 8, 32, 128), (2, 2, 16, 64)])
def test_other_shapes_keep_simd_arithmetic(indexer, shape):
    mx.random.seed(42)
    batch, width, heads, dims = shape
    q = mx.random.normal(shape).astype(mx.bfloat16)
    keys = mx.random.normal((batch, 17, dims)).astype(mx.bfloat16)
    weights = (mx.random.normal((batch, width, heads)) * 0.01).astype(mx.bfloat16)
    before, plen = [66, 47], [17, 12]
    actual = indexer.small_scores_batched(q, keys, weights, before, plen, 4)
    expected = _reference(indexer, q, keys, weights, before, plen, 4)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("width", [1, 2])
def test_zero_length_batch_row_is_fully_masked(indexer, width):
    q = mx.ones((2, width, 32, 128), dtype=mx.bfloat16)
    keys = mx.ones((2, 9, 128), dtype=mx.bfloat16)
    weights = mx.ones((2, width, 32), dtype=mx.bfloat16)
    actual = indexer.small_scores_batched(q, keys, weights, [0, 35], [0, 9], 4)
    assert mx.all(actual[0] <= -1e29).item()
    assert mx.all(actual[1] == 4096).item()
