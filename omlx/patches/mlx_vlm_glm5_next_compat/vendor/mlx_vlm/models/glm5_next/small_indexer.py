# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 DSA indexer scores for decode and verify blocks (S <= 8).

score[s, p] = sum_h w[s, h] * relu(q[s, h] . k[p]) with fp32 accumulation, in
the query dtype; incomplete or future windows get -1e30 like the general path.
"""

import mlx.core as mx

_SRC = """
    uint p = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;   // heads lane, lane+32, ...
    int P = pool_len[0];
    int qpos = before[0] + (int)s;
    float out = -1e30f;  // same fill as the general path mx.where(valid, scores, -1e30)
    if ((int)p < P && ((int)p + 1) * KPOOL - 1 <= qpos) {
        const device T* kp = k + p * HD;
        float v = 0.0f;
        for (uint h = lane; h < (uint)NH; h += 32) {
            const device T* qh = q + ((s * NH + h) * HD);
            float acc = 0.0f;
            for (int d = 0; d < HD; d += 4) {
                acc += float(qh[d]) * float(kp[d]) + float(qh[d + 1]) * float(kp[d + 1])
                     + float(qh[d + 2]) * float(kp[d + 2]) + float(qh[d + 3]) * float(kp[d + 3]);
            }
            v += metal::max(acc, 0.0f) * float(w[s * NH + h]);
        }
        out = simd_sum(v);
    }
    if (lane == 0) {
        scores[s * (uint)P + p] = static_cast<T>(out);
    }
"""

_SCORE_SIMDGROUP_LANES = 32
_SCORE_DOT_VECTOR_ELEMENTS = 4

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="glm53_dsa_indexer_small_scores",
            input_names=["q", "k", "w", "before", "pool_len"],
            output_names=["scores"],
            source=_SRC,
        )
    return _kernel


def _validate_scores(q, keys, weights, before, plen):
    if q.ndim != 4 or keys.ndim != 3 or weights.ndim != 3:
        raise ValueError("GLM small-query scorer input rank mismatch")
    batch_size, query_rows, heads, head_dim = q.shape
    if (
        min(batch_size, query_rows, heads, head_dim, keys.shape[1]) < 1
        or head_dim % _SCORE_DOT_VECTOR_ELEMENTS
        or keys.shape != (batch_size, keys.shape[1], head_dim)
        or weights.shape != (batch_size, query_rows, heads)
        or q.dtype not in (mx.float16, mx.bfloat16)
        or keys.dtype != q.dtype
        or weights.dtype != q.dtype
        or len(before) != batch_size
        or len(plen) != batch_size
        or any(p < 0 for p in before)
        or any(p < 0 or p > keys.shape[1] for p in plen)
    ):
        raise ValueError("Unsupported GLM small-query scorer inputs")


def small_scores(q, pool_keys, weights, before, kpool):
    """q (1,S,H,D), pool_keys (1,P,D), weights (1,S,H) -> (1,S,P) in q.dtype."""
    _validate_scores(q, pool_keys, weights, [before], [pool_keys.shape[1]])
    _, query_rows, query_heads, head_dim = q.shape
    pool_extent = pool_keys.shape[1]
    out = _get_kernel()(
        inputs=[
            q.reshape(-1),
            pool_keys.reshape(-1),
            weights.reshape(-1),
            mx.array([before], dtype=mx.int32),
            mx.array([pool_extent], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("NH", query_heads),
            ("HD", head_dim),
            ("KPOOL", kpool),
        ],
        grid=(pool_extent * _SCORE_SIMDGROUP_LANES, query_rows, 1),
        threadgroup=(_SCORE_SIMDGROUP_LANES, 1, 1),
        output_shapes=[(query_rows, pool_extent)],
        output_dtypes=[q.dtype],
    )[0]
    return out[None]


# Batched variant with the same arithmetic per row. rows[b] holds (pooled
# windows, tokens before) of row b; query row s sits at position before + s.

_BATCHED_SRC = """
    uint p = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;   // heads lane, lane+32, ...
    int PX = dims[0];                // pooled extent (score row stride)
    int S = dims[1];
    int P = rows[b * 2];             // this row's valid pooled windows
    int qpos = rows[b * 2 + 1] + (int)s;
    uint r = b * (uint)S + s;
    float out = -1e30f;  // same fill as the general path mx.where(valid, scores, -1e30)
    if ((int)p < P && ((int)p + 1) * KPOOL - 1 <= qpos) {
        const device T* kp = k + (size_t)b * (size_t)dims[2] + p * HD;
        float v = 0.0f;
        for (uint h = lane; h < (uint)NH; h += 32) {
            const device T* qh = q + ((r * NH + h) * HD);
            float acc = 0.0f;
            for (int d = 0; d < HD; d += 4) {
                acc += float(qh[d]) * float(kp[d]) + float(qh[d + 1]) * float(kp[d + 1])
                     + float(qh[d + 2]) * float(kp[d + 2]) + float(qh[d + 3]) * float(kp[d + 3]);
            }
            v += metal::max(acc, 0.0f) * float(w[r * NH + h]);
        }
        out = simd_sum(v);
    }
    if (lane == 0) {
        scores[r * (uint)PX + p] = static_cast<T>(out);
    }
"""

_batched_kernel = None


def _get_batched_kernel():
    global _batched_kernel
    if _batched_kernel is None:
        _batched_kernel = mx.fast.metal_kernel(
            name="glm53_dsa_indexer_small_scores_batched",
            input_names=["q", "k", "w", "rows", "dims"],
            output_names=["scores"],
            source=_BATCHED_SRC,
        )
    return _batched_kernel


_MMA_BATCHED_SRC = """
    const int P = dims[0];
    const int KSTRIDE = dims[1];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int b = int(threadgroup_position_in_grid.z);
    const int s = int(sg) / 4;
    const int h0 = (int(sg) % 4) * 8;
    const int p0 = int(threadgroup_position_in_grid.x) * 8;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    threadgroup float hs[S][32][8];
    simdgroup_matrix<float, 8, 8> acc(0.0f);
    for (int kb = 0; kb < 16; ++kb) {
        simdgroup_matrix<float, 8, 8> a, v;
        a.thread_elements()[0] = float(
            q[((b * S + s) * 32 + h0 + fm) * 128 + kb * 8 + fn]);
        a.thread_elements()[1] = float(
            q[((b * S + s) * 32 + h0 + fm) * 128 + kb * 8 + fn + 1]);
        v.thread_elements()[0] = p0 + fn < P
            ? float(k[size_t(b) * KSTRIDE + (p0 + fn) * 128 + kb * 8 + fm])
            : 0.0f;
        v.thread_elements()[1] = p0 + fn + 1 < P
            ? float(k[size_t(b) * KSTRIDE + (p0 + fn + 1) * 128 + kb * 8 + fm])
            : 0.0f;
        simdgroup_multiply_accumulate(acc, a, v, acc);
    }
    hs[s][h0 + fm][fn] = acc.thread_elements()[0];
    hs[s][h0 + fm][fn + 1] = acc.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Keep the SIMD head reduction after the MMA dot products. ReLU must
    // precede weighting because each head can have a negative weight.
    for (int j = int(sg); j < S * 8; j += S * 4) {
        const int qs = j / 8, kp = j % 8, p = p0 + kp;
        const float value = simd_sum(
            metal::max(hs[qs][lane][kp], 0.0f) * float(w[(b * S + qs) * 32 + lane]));
        if (lane == 0 && p < P) {
            bool valid = p < rows[b * 2]
                && (p + 1) * KPOOL - 1 <= rows[b * 2 + 1] + qs;
            out[(b * S + qs) * P + p] = valid ? T(value) : T(-1e30f);
        }
    }
"""

_mma_batched_kernel = None


def _get_mma_batched_kernel():
    global _mma_batched_kernel
    if _mma_batched_kernel is None:
        _mma_batched_kernel = mx.fast.metal_kernel(
            name="glm53_dsa_indexer_small_scores_mma_batched",
            input_names=["q", "k", "w", "rows", "dims"],
            output_names=["out"],
            header="#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
            source=_MMA_BATCHED_SRC,
        )
    return _mma_batched_kernel


def _pool_storage(pool_keys, pool_cache):
    """(keys, batch stride). Reads the backing buffer in place, because the
    sliced ``pooled`` view would be copied for the kernel on every call."""
    batch_size, pool_extent, head_dim = pool_keys.shape
    buf = getattr(pool_cache, "_pool_buf", None)
    if (
        buf is not None
        and getattr(pool_cache, "_pool_extent", None) == pool_extent
        and buf.ndim == 3
        and buf.shape[0] == batch_size
        and buf.shape[1] >= pool_extent
        and buf.shape[2] == head_dim
        and buf.dtype == pool_keys.dtype
    ):
        return buf, buf.shape[1] * head_dim
    return pool_keys, pool_extent * head_dim


def small_scores_batched(q, pool_keys, weights, before, plen, kpool, pool_cache=None):
    """q (B,S,H,D), pool_keys (B,P,D), weights (B,S,H) -> (B,S,P) in q.dtype."""
    _validate_scores(q, pool_keys, weights, before, plen)
    batch_size, query_rows, query_heads, head_dim = q.shape
    pool_extent = pool_keys.shape[1]
    keys, kstride = _pool_storage(pool_keys, pool_cache)
    rows = mx.array([v for pair in zip(plen, before) for v in pair], dtype=mx.int32)
    if query_rows in (1, 2) and query_heads == 32 and head_dim == 128:
        # Pack eight heads against eight keys per SIMD group; wider queries
        # and other head shapes retain the original score kernel below.
        return _get_mma_batched_kernel()(
            inputs=[q, keys, weights, rows, mx.array([pool_extent, kstride], mx.int32)],
            template=[("T", q.dtype), ("S", query_rows), ("KPOOL", kpool)],
            grid=(((pool_extent + 7) // 8) * query_rows * 128, 1, batch_size),
            threadgroup=(query_rows * 128, 1, 1),
            output_shapes=[(batch_size, query_rows, pool_extent)],
            output_dtypes=[q.dtype],
        )[0]
    return _get_batched_kernel()(
        inputs=[
            q.reshape(-1),
            keys,
            weights.reshape(-1),
            rows,
            mx.array([pool_extent, query_rows, kstride], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("NH", query_heads),
            ("HD", head_dim),
            ("KPOOL", kpool),
        ],
        grid=(pool_extent * _SCORE_SIMDGROUP_LANES, query_rows, batch_size),
        threadgroup=(_SCORE_SIMDGROUP_LANES, 1, 1),
        output_shapes=[(batch_size, query_rows, pool_extent)],
        output_dtypes=[q.dtype],
    )[0]


def batched_windows(
    indexer, q, pool_keys, weights, before, plen, native_topk, pool_cache=None
):
    """Top-k window ids (B,S,select_k) and their validity, native-selector tie order."""
    pool_extent = pool_keys.shape[1]
    select_k = min(indexer.index_topk // indexer.index_kpool, pool_extent)
    scores = small_scores_batched(
        q, pool_keys, weights, before, plen, indexer.index_kpool, pool_cache
    )
    selected = native_topk_required(native_topk, scores, select_k)
    return selected, scores


def native_topk_required(native_topk, scores, select_k):
    """A failed selector after pool mutation cannot retry the indexer."""
    try:
        selected = native_topk(scores, select_k)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "GLM small-query native top-k failed after pool update"
        ) from exc
    if selected is None:
        raise RuntimeError("GLM small-query native top-k unavailable after pool update")
    if selected.shape != scores.shape[:-1] + (select_k,):
        raise RuntimeError("GLM small-query native top-k returned an invalid shape")
    return selected.astype(mx.int32)


def expand_windows(indexer, selected, before, plen, left_padding=None):
    """Expand native window order, causal validity and incomplete-window tails."""
    batch_size, query_rows, selected_count = selected.shape
    kp = indexer.index_kpool
    before = before if isinstance(before, list) else [before] * batch_size
    plen = plen if isinstance(plen, list) else [plen] * batch_size
    qpos = mx.array(before, dtype=mx.int32)[:, None] + mx.arange(query_rows)[None]
    lengths = mx.array(plen, dtype=mx.int32)[:, None, None]
    lp = (
        mx.zeros((batch_size,), mx.int32)
        if left_padding is None
        else left_padding.astype(mx.int32)
    )
    valid = (
        (selected >= 0)
        & (selected < lengths)
        & ((selected + 1) * kp - 1 <= qpos[..., None])
    )
    indices = selected[..., None] * kp + mx.arange(kp) + lp[:, None, None, None]
    indices = mx.where(valid[..., None], indices, -1).reshape(
        batch_size, query_rows, selected_count * kp
    )
    tail_on = indexer.index_kpool_always_select_tail and kp > 1
    width = indexer.index_topk + (kp - 1 if tail_on else 0)
    if tail_on:
        count = (qpos + 1) % kp
        start = qpos + 1 - count
        off = mx.arange(kp - 1)
        tail = mx.where(
            off < count[..., None], start[..., None] + off + lp[:, None, None], -1
        )
        indices = mx.concatenate([indices, tail.astype(mx.int32)], axis=-1)
    if indices.shape[-1] < width:
        indices = mx.concatenate(
            [
                indices,
                mx.full(
                    (batch_size, query_rows, width - indices.shape[-1]), -1, mx.int32
                ),
            ],
            axis=-1,
        )
    return indices[..., :width][:, None].astype(mx.int32)
