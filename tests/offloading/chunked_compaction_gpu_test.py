# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lifted stream compaction runs on the GPU as per-block counts and ordered per-block writes: one thread walks one
block, so no buffer is as long as the iteration space."""

import numpy as np
import pytest
from host_library_calls_test import gpu_final, outer_maps

import dace
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import IDX_PREFIX, MASK_PREFIX
from dace.transformation.passes.cpu_specialization.chunk_stream_compaction import GPU_BLOCK

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def compact_scaled_gpu(src: dace.float64[N], weight: dace.float64[N], packed: dace.float64[N], count: dace.int64[1]):
    j = 0
    for i in range(N):
        if src[i] > 0.0:
            packed[j] = src[i] * weight[i]
            j = j + 1
    count[0] = j


def test_the_gpu_form_has_block_kernels_and_no_mask_or_rank_buffer():
    sdfg = gpu_final("chunked_compaction_gpu_structure", compact_scaled_gpu)
    assert not any(name.startswith((MASK_PREFIX, IDX_PREFIX)) for _, name, _ in sdfg.arrays_recursive())
    assert any(name.startswith("compaction_counts") for name in sdfg.arrays)
    # A count and a write map per compaction, of which there is one per index width
    blocks = [e for e in outer_maps(sdfg) if e.map.params == ["block"]]
    assert blocks and all(e.map.schedule == dace.ScheduleType.GPU_Device for e in blocks)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, GPU_BLOCK, 5 * GPU_BLOCK + 11])
def test_the_gpu_form_packs_in_order_and_counts(n: int):
    import cupy

    sdfg = gpu_final(f"chunked_compaction_gpu_{n}", compact_scaled_gpu)
    rng = np.random.default_rng(n)
    src = rng.standard_normal(n)
    weight = rng.standard_normal(n)
    packed = cupy.zeros(n)
    count = cupy.zeros(1, dtype=np.int64)
    sdfg(src=cupy.asarray(src), weight=cupy.asarray(weight), packed=packed, count=count, N=n)
    want = src[src > 0.0] * weight[src > 0.0]
    assert int(count[0]) == want.size
    np.testing.assert_array_equal(cupy.asnumpy(packed)[: want.size], want)


if __name__ == "__main__":
    test_the_gpu_form_has_block_kernels_and_no_mask_or_rank_buffer()
    test_the_gpu_form_packs_in_order_and_counts(1)
    test_the_gpu_form_packs_in_order_and_counts(7)
    test_the_gpu_form_packs_in_order_and_counts(GPU_BLOCK)
    test_the_gpu_form_packs_in_order_and_counts(5 * GPU_BLOCK + 11)
