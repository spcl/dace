# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A device-to-host copy of a vectorized kernel's result must wait for the kernel.

CloudSC's flux kernel writes ``pfsqrf[jk, jl]``, reads it back and writes it again. Nesting the tiled
column body binds ``pfsqrf`` through two connectors -- an access node inside the kernel for the
reread and the MapExit for the final write -- and ``ExpandNestedSDFGInputs`` folded them onto the
in-kernel one. The kernel's write then never left the kernel, the host copy of ``pfsqrf`` lost its
producer, and the generated code copied ``pfsqrf`` out before launching the kernel: every row came
back as its input value.
"""
import numpy as np
import pytest

import dace
from dace.config import set_temporary
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

ROWS, COLS = 8, 16


def flux_reread_kernel() -> dace.SDFG:
    """The offloaded CloudSC shape: host arrays mirrored by ``gpu_`` transients, a ``GPU_Device`` map
    over rows around a ``Sequential`` column map, and the copy out in the kernel's state. The result
    node is added first, so a copy that lost its producer is a source and is emitted first."""
    sdfg = dace.SDFG('flux_reread_copy_out')
    for name in ('lf', 'rf'):
        sdfg.add_array(name, [ROWS + 1, COLS], dace.float64)
        sdfg.add_array(f'gpu_{name}', [ROWS + 1, COLS],
                       dace.float64,
                       storage=dace.StorageType.GPU_Global,
                       transient=True)
    sdfg.add_scalar('s', dace.float64, transient=True, storage=dace.StorageType.Register)
    st = sdfg.add_state('main')
    result = st.add_access('gpu_rf')
    host_rf = st.add_write('rf')
    device_lf = st.add_access('gpu_lf')
    st.add_nedge(st.add_read('lf'), device_lf, dace.Memlet(f'lf[0:{ROWS + 1}, 0:{COLS}]'))
    ome, omx = st.add_map('rows', {'jk': f'1:{ROWS + 1}'}, schedule=dace.ScheduleType.GPU_Device)
    ome.map.gpu_block_size = [ROWS, 1, 1]
    ime, imx = st.add_map('cols', {'jl': f'0:{COLS}'}, schedule=dace.ScheduleType.Sequential)
    first = st.add_tasklet('first', {'_in': None}, {'_out': None}, '_out = _in')
    mid = st.add_access('gpu_rf')
    reread = st.add_tasklet('reread', {'_in': None}, {'_out': None}, '_out = _in')
    scalar = st.add_access('s')
    update = st.add_tasklet('update', {'_in': None}, {'_out': None}, '_out = _in * 2.0 + 1.0')
    st.add_memlet_path(device_lf, ome, ime, first, dst_conn='_in', memlet=dace.Memlet('gpu_lf[jk - 1, jl]'))
    st.add_edge(first, '_out', mid, None, dace.Memlet('gpu_rf[jk, jl]'))
    st.add_edge(mid, None, reread, '_in', dace.Memlet('gpu_rf[jk, jl]'))
    st.add_edge(reread, '_out', scalar, None, dace.Memlet('s[0]'))
    st.add_edge(scalar, None, update, '_in', dace.Memlet('s[0]'))
    st.add_memlet_path(update, imx, omx, result, src_conn='_out', memlet=dace.Memlet('gpu_rf[jk, jl]'))
    st.add_nedge(result, host_rf, dace.Memlet(f'gpu_rf[1:{ROWS + 1}, 0:{COLS}]',
                                              other_subset=f'1:{ROWS + 1}, 0:{COLS}'))
    sdfg.validate()
    return sdfg


def test_the_vectorized_kernel_write_still_feeds_the_copy_out():
    sdfg = flux_reread_kernel()
    VectorizeGPU(VectorizeConfig(widths=(2, ), validate=True)).apply_pass(sdfg, {})
    copies = [(state, n) for state in sdfg.states() for n in state.data_nodes() if n.data == 'gpu_rf' and any(
        isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'rf' for e in state.out_edges(n))]
    assert len(copies) == 1, copies
    state, source = copies[0]
    producers = [e.src for e in state.in_edges(source)]
    assert any(isinstance(p, nodes.MapExit) and p.map.label == 'rows' for p in producers), producers


@pytest.mark.gpu
def test_the_vectorized_kernel_result_reaches_the_host():
    sdfg = flux_reread_kernel()
    VectorizeGPU(VectorizeConfig(widths=(2, ), validate=True)).apply_pass(sdfg, {})
    rng = np.random.default_rng(0)
    lf = rng.random((ROWS + 1, COLS))
    rf = np.zeros((ROWS + 1, COLS))
    with set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        sdfg(lf=lf, rf=rf)
    want = lf[:-1] * 2.0 + 1.0
    assert np.allclose(rf[1:], want, rtol=0, atol=1e-14), f"max|diff|={np.abs(rf[1:] - want).max():.3e}"
