# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WarpTiling and fusion on the softmax operator. """
import dace
from dace.transformation.dataflow import (MapFusion, WarpTiling,
                                          TrivialMapElimination, Vectorization)
from dace.transformation.interstate import (HoistState, InlineSDFG, StateFusion,
                                            GPUTransformSDFG)
from dace.transformation.subgraph import (SubgraphFusion, MultiExpansion,
                                          ReduceExpansion)

import numpy as np
import pytest

dn1, dn2, dn3, dr = (dace.symbol(s) for s in ('dn1', 'dn2', 'dn3', 'dr'))


@dace.program
def softmax_fwd(inp: dace.float32[dn1, dn2, dn3, dr],
                out: dace.float32[dn1, dn2, dn3, dr]):
    max = np.max(inp, axis=-1)
    max_keepdims = np.reshape(max, (dn1, dn2, dn3, 1))
    exp_arr = np.exp(inp - max_keepdims)
    sum = np.sum(exp_arr, axis=-1)
    sum_keepdims = np.reshape(sum, (dn1, dn2, dn3, 1))
    out[:] = exp_arr / sum_keepdims


# Numerically-stable version of softmax
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


@pytest.mark.gpu
def test_warp_softmax(vector_length=1):
    # Get SDFG
    sdfg = softmax_fwd.to_sdfg(coarsen=True)

    # Apply transformations
    sdfg.apply_transformations_repeated(ReduceExpansion)
    MultiExpansion.apply_to(sdfg, sdfg.node(0).nodes())
    SubgraphFusion.apply_to(sdfg, sdfg.node(0).nodes())
    sdfg.expand_library_nodes()
    sdfg.coarsen_dataflow()
    sdfg.apply_transformations_repeated([TrivialMapElimination, MapFusion])
    sdfg.apply_transformations(GPUTransformSDFG)
    assert sdfg.apply_transformations(WarpTiling) == 1
    sdfg.apply_transformations_repeated([HoistState, InlineSDFG, StateFusion])
    sdfg.apply_transformations_repeated([TrivialMapElimination, MapFusion])
    if vector_length != 1:
        sdfg.apply_transformations_repeated(
            Vectorization,
            dict(vector_len=vector_length,
                 preamble=False,
                 postamble=False,
                 strided_map=False))
    sdfg.specialize(dict(dn1=2, dn2=16, dn3=128, dr=128))

    # Check validity
    sdfg.validate()
    assert sdfg.number_of_nodes() == 1
    state = sdfg.node(0)
    assert len([
        c for c in state.scope_children()[None]
        if isinstance(c, dace.nodes.MapEntry)
    ]) == 1

    # Check correctness
    inp = np.random.rand(2, 16, 128, 128).astype(np.float32)
    out = np.random.rand(2, 16, 128, 128).astype(np.float32)
    reg_out = softmax(inp)

    sdfg(inp=inp, out=out)

    assert np.allclose(out, reg_out, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    test_warp_softmax()
