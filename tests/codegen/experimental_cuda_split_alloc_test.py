# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test for the experimental CUDA backend's split-DECLARE/ALLOCATE path.

When a Scope-lifetime GPU transient has a non-free-symbolic shape and is used
across multiple states, the framecode splits its allocation: ``declare_array``
emits the ``T*`` declaration at SDFG scope, and ``allocate_array`` emits the
``cudaMalloc`` in the first producing state. The host pointer name must end up
in ``defined_vars`` at a scope that survives state boundaries, otherwise the
consuming state's kernel codegen (``_define_variables_in_kernel_scope``) fails
with ``KeyError: 'Variable X has not been defined'``.

The reproducer hand-builds that exact SDFG shape, so the test does not depend
on any frontend or transformation pass.
"""
import pytest

import dace
from dace.sdfg.state import LoopRegion


def _build_split_scope_transient_sdfg():
    L = dace.symbol('L', dace.int64)
    length_sym = dace.symbol('length', dace.int64)
    GPU = dace.dtypes.StorageType.GPU_Global

    sdfg = dace.SDFG('split_scope_lifetime_transient')
    sdfg.add_symbol('L', dace.int64)
    sdfg.add_symbol('length', dace.int64)
    sdfg.add_array('Z', (L, ), dace.float64, storage=GPU)
    sdfg.add_array('C', (L, ), dace.float64, storage=GPU)
    sdfg.add_array('out', (L, ), dace.float64, storage=GPU)
    # The shape uses ``length``, which is assigned by the LoopRegion -- not a
    # free SDFG symbol. ``is_nonfree_sym_dependent`` therefore returns True,
    # triggering the split DECLARE/ALLOCATE codegen path.
    sdfg.add_transient('tmp', (length_sym, ), dace.float64, storage=GPU, lifetime=dace.dtypes.AllocationLifetime.Scope)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion(label='lr',
                      condition_expr='length > 0',
                      loop_var='length',
                      initialize_expr='length = L',
                      update_expr='length = length - 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())

    write_tmp = loop.add_state('write_tmp', is_start_block=True)
    z_in = write_tmp.add_read('Z')
    tmp_w = write_tmp.add_write('tmp')
    me, mx = write_tmp.add_map('mul_map', dict(i='0:length'), schedule=dace.ScheduleType.GPU_Device)
    t = write_tmp.add_tasklet('mul', {'a'}, {'b'}, 'b = a * a')
    write_tmp.add_memlet_path(z_in, me, t, dst_conn='a', memlet=dace.Memlet('Z[i]'))
    write_tmp.add_memlet_path(t, mx, tmp_w, src_conn='b', memlet=dace.Memlet('tmp[i]'))

    read_tmp = loop.add_state('read_tmp')
    tmp_r = read_tmp.add_read('tmp')
    c_in = read_tmp.add_read('C')
    o_w = read_tmp.add_write('out')
    me2, mx2 = read_tmp.add_map('add_map', dict(i='0:length'), schedule=dace.ScheduleType.GPU_Device)
    t2 = read_tmp.add_tasklet('add', {'a', 'c'}, {'b'}, 'b = a + c')
    read_tmp.add_memlet_path(tmp_r, me2, t2, dst_conn='a', memlet=dace.Memlet('tmp[i]'))
    read_tmp.add_memlet_path(c_in, me2, t2, dst_conn='c', memlet=dace.Memlet('C[i]'))
    read_tmp.add_memlet_path(t2, mx2, o_w, src_conn='b', memlet=dace.Memlet('out[i]'))

    loop.add_edge(write_tmp, read_tmp, dace.InterstateEdge())
    return sdfg


@pytest.mark.gpu
def test_split_scope_lifetime_transient_across_states():
    # Bug surfaces only on the experimental CUDA backend; force-select it.
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        sdfg = _build_split_scope_transient_sdfg()
        sdfg.validate()
        # The compile path is the regression site (codegen-time KeyError).
        sdfg.compile()


if __name__ == '__main__':
    test_split_scope_lifetime_transient_across_states()
