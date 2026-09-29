# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WarpTiling and fusion on the softmax operator. """
import numpy as np
import pytest

import dace
from dace.transformation.dataflow import (MapFusionVertical, ReduceExpansion, TrivialMapElimination, Vectorization,
                                          WarpTiling)
from dace.transformation.interstate import HoistState, InlineSDFG, StateFusion
from dace.transformation.subgraph import MultiExpansion, SubgraphFusion

dn1, dn2, dn3, dr = (dace.symbol(s) for s in ('dn1', 'dn2', 'dn3', 'dr'))


@dace.program
def softmax_fwd(inp: dace.float32[dn1, dn2, dn3, dr], out: dace.float32[dn1, dn2, dn3, dr]):
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
    sdfg = softmax_fwd.to_sdfg(simplify=True)

    # Apply transformations
    sdfg.apply_transformations_repeated(ReduceExpansion, validate_all=True)
    MultiExpansion.apply_to(sdfg, sdfg.node(0).nodes())
    SubgraphFusion.apply_to(sdfg, sdfg.node(0).nodes())
    sdfg.expand_library_nodes()
    sdfg.simplify()
    sdfg.apply_transformations_repeated([TrivialMapElimination, MapFusionVertical], validate_all=True)
    sdfg.apply_gpu_transformations(validate_all=True, simplify=False)
    assert sdfg.apply_transformations(WarpTiling) == 1
    sdfg.apply_transformations_repeated([HoistState, InlineSDFG, StateFusion], validate_all=True)
    sdfg.apply_transformations_repeated([TrivialMapElimination, MapFusionVertical], validate_all=True)
    if vector_length != 1:
        sdfg.apply_transformations_repeated(Vectorization,
                                            dict(vector_len=vector_length,
                                                 preamble=False,
                                                 postamble=False,
                                                 strided_map=False),
                                            validate_all=True)
    sdfg.specialize(dict(dn1=2, dn2=16, dn3=128, dr=128))

    # Check validity
    sdfg.validate()
    assert sdfg.number_of_nodes() == 1
    state = sdfg.node(0)
    assert len([c for c in state.scope_children()[None] if isinstance(c, dace.nodes.MapEntry)]) == 1

    # Check correctness
    inp = np.random.rand(2, 16, 128, 128).astype(np.float32)
    out = np.random.rand(2, 16, 128, 128).astype(np.float32)
    reg_out = softmax(inp)

    sdfg(inp=inp, out=out)

    assert np.allclose(out, reg_out, rtol=1e-4, atol=1e-6)


def warp_matvec_plus() -> dace.SDFG:
    """``y[i] = b[i] + A[i, :] @ x`` with a register accumulator in the kernel body itself (not nested)."""
    sdfg = dace.SDFG('warp_matvec_plus')
    for name, shape in (('A', ['M', 'N']), ('x', ['N']), ('b', ['M']), ('y', ['M'])):
        sdfg.add_array(name, shape, dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_scalar('acc', dace.float64, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    rows, rows_exit = state.add_map('rows', dict(i='0:M'), schedule=dace.ScheduleType.GPU_Device)
    cols, cols_exit = state.add_map('cols', dict(j='0:N'))
    A, x, b, y = (state.add_access(name) for name in 'Axby')
    acc_init, acc = state.add_access('acc'), state.add_access('acc')
    init = state.add_tasklet('init', {'inp'}, {'out'}, 'out = inp')
    mult = state.add_tasklet('mult', {'a', 'v'}, {'out'}, 'out = a * v')
    store = state.add_tasklet('store', {'inp'}, {'out'}, 'out = inp')
    state.add_memlet_path(b, rows, init, dst_conn='inp', memlet=dace.Memlet('b[i]'))
    state.add_edge(init, 'out', acc_init, None, dace.Memlet('acc[0]'))
    state.add_memlet_path(A, rows, cols, mult, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(x, rows, cols, mult, dst_conn='v', memlet=dace.Memlet('x[j]'))
    state.add_nedge(acc_init, cols, dace.Memlet())
    state.add_memlet_path(mult, cols_exit, acc, src_conn='out', memlet=dace.Memlet('acc[0]', wcr='lambda a, b: a + b'))
    state.add_edge(acc, None, store, 'inp', dace.Memlet('acc[0]'))
    state.add_memlet_path(store, rows_exit, y, src_conn='out', memlet=dace.Memlet('y[i]'))
    sdfg.validate()
    return sdfg, rows


@pytest.mark.gpu
def test_warp_reduction_counts_the_initial_value_once():
    """Each lane's partial starts at the identity, so ``b[i]`` is added once, not once per lane."""
    import cupy as cp
    sdfg, rows = warp_matvec_plus()
    # A ``Default`` inner map reads as an explicit thread-block map to ``can_be_applied``; it is the one to stride.
    WarpTiling.apply_to(sdfg, mapentry=rows, verify=False)
    sdfg.validate()
    seeds = [
        node for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.Tasklet) and node.label == 'lane_partial_seed'
    ]
    assert len(seeds) == 1 and seeds[0].code.as_string.strip() == '__out = double(0.0);'

    m, n = 37, 300
    rng = np.random.default_rng(0)
    A, x, b = rng.random((m, n)), rng.random(n), rng.random(m)
    y = cp.zeros(m)
    sdfg(A=cp.asarray(A), x=cp.asarray(x), b=cp.asarray(b), y=y, M=m, N=n)
    np.testing.assert_allclose(y.get(), b + A @ x, rtol=1e-12)


if __name__ == '__main__':
    test_warp_softmax()
    test_warp_reduction_counts_the_initial_value_once()
