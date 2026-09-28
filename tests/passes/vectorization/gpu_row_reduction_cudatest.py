# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A per-row reduction inside a host loop survives the GPU vectorizer (CloudSC fluxes).

Canonicalizing for the GPU lowers ``acc += a[jl, jm]`` to a ``Reduce`` over a row view ``x[0:M]`` and a
copy of the result into a length-1 output ``y``, both inside a nested SDFG. Widening that nested SDFG's
boundary to the outer arrays used to keep ``Reduce.axes == [0]`` (now the length-1 row dim of
``a[jl, 0:M]``, so the reduce became a copy) and to leave the copy's implicit destination at ``s[0]``.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

L, M = dace.symbol('L'), dace.symbol('M')


@dace.program
def row_sums(a: dace.float64[L, M], s: dace.float64[L]):
    for jl in dace.map[0:L]:
        acc = 0.0
        for jm in range(M):
            acc = acc + a[jl, jm]
        s[jl] = acc


def vectorized_row_sums() -> dace.SDFG:
    sdfg = row_sums.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target='gpu')
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(2, ))).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_the_reduce_keeps_reducing_the_row_after_widening():
    sdfg = vectorized_row_sums()
    reduces = [(n, st) for n, st in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert reduces
    for node, state in reduces:
        (edge, ) = state.in_edges(node)
        extents = edge.data.subset.size()
        assert all(extents[axis] != 1 for axis in node.axes), (str(edge.data), node.axes)


def test_the_result_copy_lands_on_its_row():
    sdfg = vectorized_row_sums()
    copies = [
        e.data for e, _ in sdfg.all_edges_recursive() if isinstance(e.data, dace.Memlet)
        and isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == 's' and isinstance(e.src, dace.nodes.AccessNode)
    ]
    assert copies
    assert all(m.other_subset is not None and m.other_subset.free_symbols for m in copies), [str(m) for m in copies]


@pytest.mark.gpu
def test_row_sums_on_the_device():
    import cupy
    sdfg = vectorized_row_sums()
    a = np.random.default_rng(0).random((33, 5))
    s = cupy.zeros(33)
    sdfg(a=cupy.asarray(a), s=s, L=33, M=5)
    np.testing.assert_allclose(s.get(), a.sum(axis=1), rtol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
