# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ChunkStreamCompaction: a lifted stream compaction runs as per-block counts and ordered per-block writes on the CPU,
with the same results and no mask or rank buffer."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.nodes.scan import Scan
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import IDX_PREFIX, MASK_PREFIX
from dace.transformation.passes.cpu_specialization.chunk_stream_compaction import BLOCK

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def compact_scaled(src: dace.float64[N], weight: dace.float64[N], packed: dace.float64[N], count: dace.int64[1]):
    j = 0
    for i in range(N):
        if src[i] > 0.0:
            packed[j] = src[i] * weight[i]
            j = j + 1
    count[0] = j


def lowered() -> dace.SDFG:
    sdfg = compact_scaled.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    finalize_for_target(sdfg, "cpu", validate=True)
    return sdfg


def test_the_chunked_form_has_no_mask_rank_scan_or_reduce():
    sdfg = lowered()
    assert not any(name.startswith((MASK_PREFIX, IDX_PREFIX)) for name in sdfg.arrays)
    assert not any(isinstance(node, (Scan, Reduce)) for node, _ in sdfg.all_nodes_recursive())
    assert any(name.startswith("compaction_counts") for name in sdfg.arrays)


@pytest.mark.parametrize("n", [1, 7, BLOCK - 1, BLOCK, 3 * BLOCK + 11])
def test_the_chunked_form_packs_in_order_and_counts(n: int):
    sdfg = lowered()
    rng = np.random.default_rng(n)
    src = rng.standard_normal(n)
    weight = rng.standard_normal(n)
    packed = np.zeros(n)
    count = np.zeros(1, dtype=np.int64)
    sdfg(src=src, weight=weight, packed=packed, count=count, N=n)
    kept = src > 0.0
    want = src[kept] * weight[kept]
    assert int(count[0]) == want.size
    np.testing.assert_array_equal(packed[: want.size], want)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
