# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel-local buffer that control flow reads, lifted out of the kernel.

``MoveArrayOutOfKernel`` gives a kernel-local buffer one slice per kernel iteration, so every
subscript of it grows a leading index. Memlets and inlined tasklet bodies were rewritten; an
interstate-edge assignment that reads one element of the buffer (``sel = order[0]``) was not, so
the stale rank-1 subscript ended up naming a whole row of the rank-2 buffer. Code generation
rejects that with ``SyntaxError: Range subscripts disallowed in interstate edges``, the error
npbench ``rayleigh_ritz_rotation`` hit on the GPU canonicalize column.

Built by hand: the frontend does not produce a kernel-local buffer that an interstate edge reads.
"""
import ast

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUCodegenPreprocessPipeline

NX, NZ = (dace.symbol(s, dtype=dace.int64) for s in ('NX', 'NZ'))


def kernel_with_interstate_buffer_read() -> dace.SDFG:
    """``out[i] = a[i, order[i, 0]]`` where ``order`` is a symbolically-sized kernel-local buffer.

    ``order`` is filled from ``a`` (a constant fill is folded away, and the buffer with it) and
    then read by the interstate edge that carries ``sel`` into the consuming state. The symbolic
    extent is what forces the lift: such a buffer has no device-local form (a VLA in device code)
    and no per-thread register form either.
    """
    inner = dace.SDFG('pick_body')
    inner.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    inner.add_array('out', [NX], dace.float64, storage=dtypes.StorageType.GPU_Global)
    inner.add_array('order', [NZ], dace.int64, transient=True, storage=dtypes.StorageType.Register)
    inner.add_symbol('sel', dace.int64)

    fill = inner.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('rank', {'k': '0:NZ'}, {'__in': dace.Memlet('a[i, k]')},
                            '__out = (NZ - 1 - k) if (__in > 0.5) else k', {'__out': dace.Memlet('order[k]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)

    use = inner.add_state('use')
    inner.add_edge(fill, use, dace.InterstateEdge(assignments={'sel': 'order[0]'}))
    use.add_edge(use.add_read('a'), None, use.add_tasklet('pick', {'__in'}, {'__out'}, '__out = __in'), '__in',
                 dace.Memlet('a[i, sel]'))
    pick = next(node for node in use.nodes() if isinstance(node, dace.nodes.Tasklet))
    use.add_edge(pick, '__out', use.add_write('out'), None, dace.Memlet('out[i]'))

    sdfg = dace.SDFG('kernel_with_interstate_buffer_read')
    sdfg.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [NX], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('body', is_start_block=True)
    entry, exit_node = state.add_map('grid', dict(i='0:NX'), schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {'a'}, {'out'}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    state.add_memlet_path(state.add_read('a'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('a[0:NX, 0:NZ]'))
    state.add_memlet_path(nsdfg, exit_node, state.add_write('out'), src_conn='out', memlet=dace.Memlet('out[0:NX]'))
    sdfg.validate()
    return sdfg


def buffer_reads_on_interstate_edges(sdfg: dace.SDFG):
    """Every ``(owning SDFG, subscript AST)`` pair by which an interstate edge reads ``order``."""
    reads = []
    for nested in sdfg.all_sdfgs_recursive():
        for edge in nested.all_interstate_edges():
            for value in edge.data.assignments.values():
                for node in ast.walk(ast.parse(str(value))):
                    if isinstance(node, ast.Subscript) and getattr(node.value, 'id', None) == 'order':
                        reads.append((nested, node))
    return reads


def test_lifted_buffer_interstate_read_gains_the_kernel_index():
    """The read must keep naming one element: as many indices as the lifted buffer has dimensions."""
    sdfg = kernel_with_interstate_buffer_read()
    GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})

    reads = buffer_reads_on_interstate_edges(sdfg)
    assert reads, 'the interstate read of the buffer vanished'
    for owner, subscript in reads:
        rank = len(owner.arrays['order'].shape)
        assert rank == 2, f'the buffer was not lifted: shape {owner.arrays["order"].shape}'
        indices = subscript.slice.elts if isinstance(subscript.slice, ast.Tuple) else [subscript.slice]
        assert not any(isinstance(index, ast.Slice) for index in indices), ast.unparse(subscript)
        assert len(indices) == rank, f'{ast.unparse(subscript)} reads a rank-{rank} buffer'


def test_lifted_buffer_interstate_read_generates_code():
    """The defect surfaced in code generation: a range subscript on an interstate edge stops it."""
    sdfg = kernel_with_interstate_buffer_read()
    GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})
    sdfg.generate_code()


@pytest.mark.gpu
def test_lifted_buffer_interstate_read_computes_the_right_values():
    """The structural checks miss a stale index that stays rank-correct; it compiles and returns garbage."""
    cupy = pytest.importorskip('cupy')

    nx, nz = 5, 7
    host_a = np.random.default_rng(0).random((nx, nz))
    order_first = np.where(host_a[:, 0] > 0.5, nz - 1, 0)
    expected = host_a[np.arange(nx), order_first]

    out = cupy.zeros(nx)
    kernel_with_interstate_buffer_read()(a=cupy.asarray(host_a), out=out, NX=nx, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), expected)


if __name__ == '__main__':
    test_lifted_buffer_interstate_read_gains_the_kernel_index()
    test_lifted_buffer_interstate_read_generates_code()
    test_lifted_buffer_interstate_read_computes_the_right_values()
