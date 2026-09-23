# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A container the device writes before a guarded specialization is copied back for a host read after it.

The location propagation walked the IR in pre-order, so the close node of a conditional passed its
locations on once its FIRST arm had reached it. A guarded specialization lists its fallback arm first,
and that arm pushes nothing into the close node, so the blocks after the specialization never learnt
where the container lived. A host read after it was renamed to a host twin no copy ever declared
(ls3df_scf's ``b_frag`` failed the canon GPU column this way), and what the parallel arm wrote on the
device was never copied back to the caller, which got its input back unchanged.
"""
import numpy as np
import pytest

import dace
from dace import InterstateEdge, Memlet
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol('N')
#: Below this extent the specialization takes its sequential fallback arm.
SMALL = 4


def device_write_specialization_host_read() -> dace.SDFG:
    """``B = 2 * A`` in a kernel, then ``C += A`` guarded (pinned loop / map), then ``out = B[0]`` on the host."""
    sdfg = dace.SDFG('device_write_specialization_host_read')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('C', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_transient('B', [N], dace.float64)

    write = sdfg.add_state('write_B', is_start_block=True)
    write.add_mapped_tasklet('double', {'i': '0:N'}, {'a': Memlet('A[i]')},
                             'b = 2 * a', {'b': Memlet('B[i]')},
                             external_edges=True)

    dispatch = ConditionalBlock('dispatch')
    sdfg.add_node(dispatch)
    sdfg.add_edge(write, dispatch, InterstateEdge())
    fallback = ControlFlowRegion('fallback', sdfg=sdfg)
    dispatch.add_branch(CodeBlock(f'N < {SMALL}'), fallback)
    loop = LoopRegion('sequential', 'j < N', 'j', 'j = 0', 'j = j + 1')
    loop.pinned_sequential = True
    fallback.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    accumulate = body.add_tasklet('accumulate', {'a': None, 'c_in': None}, {'c': None}, 'c = c_in + a')
    body.add_edge(body.add_read('A'), None, accumulate, 'a', Memlet('A[j]'))
    body.add_edge(body.add_read('C'), None, accumulate, 'c_in', Memlet('C[j]'))
    body.add_edge(accumulate, 'c', body.add_write('C'), None, Memlet('C[j]'))
    parallel = ControlFlowRegion('parallel', sdfg=sdfg)
    dispatch.add_branch(None, parallel)
    parallel.add_state('map', is_start_block=True).add_mapped_tasklet('accumulate', {'i': '0:N'}, {
        'a': Memlet('A[i]'),
        'c_in': Memlet('C[i]')
    },
                                                                      'c = c_in + a', {'c': Memlet('C[i]')},
                                                                      external_edges=True)

    read = sdfg.add_state('read_B')
    sdfg.add_edge(dispatch, read, InterstateEdge())
    first = read.add_tasklet('first', {'b': None}, {'o': None}, 'o = b')
    read.add_edge(read.add_read('B'), None, first, 'b', Memlet('B[0]'))
    read.add_edge(first, 'o', read.add_write('out'), None, Memlet('out[0]'))
    return sdfg


def offloaded() -> dace.SDFG:
    sdfg = device_write_specialization_host_read()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def top_level_copies(sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """``(source, destination)`` of every copy in a state of the SDFG's own top level."""
    return [(src.data, dst.data) for state in sdfg.nodes() if isinstance(state, dace.SDFGState)
            for src in state.data_nodes() for dst in state.successors(src) if isinstance(dst, dace.nodes.AccessNode)]


def test_the_blocks_after_a_specialization_know_where_its_data_lives():
    """Every container the offloaded graph names is declared, ``B`` crosses to the host for the read,
    and what the parallel arm wrote on the device goes back to the caller's ``C``."""
    sdfg = offloaded()
    undeclared = {(state.label, node.data)
                  for state in sdfg.all_states()
                  for node in state.data_nodes() if node.data not in sdfg.arrays}
    assert not undeclared, undeclared
    read_state = next(state for state in sdfg.all_states() if state.label == 'read_B')
    (read_B, ) = [node for node in read_state.data_nodes() if node.data.startswith('B')]
    assert sdfg.arrays[read_B.data].storage != dace.StorageType.GPU_Global
    copies = top_level_copies(sdfg)
    assert ('B', read_B.data) in copies, copies
    assert ('C_gpu', 'C') in copies, copies


@pytest.mark.gpu
@pytest.mark.parametrize('n', [SMALL - 2, 2 * SMALL])
def test_the_offloaded_specialization_computes_what_numpy_computes(n):
    """Both arms: the fallback below ``SMALL``, the map above it."""
    sdfg = offloaded()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    A = np.arange(1, n + 1, dtype=np.float64)
    C = np.full(n, 10.0)
    out = np.zeros(1)
    sdfg(A=A, C=C, out=out, N=n)
    np.testing.assert_array_equal(C, A + 10.0)
    np.testing.assert_array_equal(out, [2 * A[0]])
