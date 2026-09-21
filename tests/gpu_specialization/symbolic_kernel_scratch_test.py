# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A symbolically-sized scratch buffer inside a GPU kernel, end to end.

Such a buffer has no device-local form: emitted as a stack array it is a VLA, and nvcc refuses one
("expression must have a constant value") where the host compiler accepts it. It has no per-thread
register form either, its extent being unknown at compile time. The only lowering left is the one
``MoveArrayOutOfKernel`` provides -- one slice of a device-global buffer per kernel iteration -- so
this pins that the promotion happens AND that the resulting program computes the right thing.

Built by hand: the frontend's fusion recomputes such a buffer away, and the buffer is the subject.
"""
import numpy as np
import pytest

import dace
from dace import dtypes

NX, NY, NZ = (dace.symbol(s, dtype=dace.int64) for s in ('NX', 'NY', 'NZ'))


def kernel_with_symbolic_scratch(halo: int = 0) -> dace.SDFG:
    """``out[i, j, k] = 2 * a[i, j, NZ - 1 - k] + 1`` through a per-iteration ``tmp[NZ]``.

    The reversed read is what keeps ``tmp`` alive: a straight-through copy is recomputed into the
    consumer and the buffer disappears. ``halo`` leaves that many boundary planes of ``out``
    unwritten, so the kernel map starts at ``halo`` the way a stencil's interior sweep does.
    """
    inner = dace.SDFG('scratch_body')
    inner.add_array('a', [NX, NY, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    inner.add_array('out', [NX, NY, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    inner.add_array('tmp', [NZ], dace.float64, transient=True, storage=dtypes.StorageType.Register)

    fill = inner.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('scale', {'k': '0:NZ'}, {'__in': dace.Memlet('a[i, j, k]')},
                            '__out = __in * 2.0', {'__out': dace.Memlet('tmp[k]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)
    drain = inner.add_state_after(fill, 'drain')
    drain.add_mapped_tasklet('shift', {'k': '0:NZ'}, {'__in': dace.Memlet('tmp[NZ - 1 - k]')},
                             '__out = __in + 1.0', {'__out': dace.Memlet('out[i, j, k]')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)

    sdfg = dace.SDFG(f'kernel_with_symbolic_scratch_halo{halo}')
    sdfg.add_array('a', [NX, NY, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [NX, NY, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('body', is_start_block=True)
    entry, exit_node = state.add_map('grid',
                                     dict(i=f'{halo}:NX-{halo}', j=f'{halo}:NY-{halo}'),
                                     schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {'a'}, {'out'}, symbol_mapping=dict(i='i', j='j', NX=NX, NY=NY, NZ=NZ))
    state.add_memlet_path(state.add_read('a'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('a[0:NX, 0:NY, 0:NZ]'))
    state.add_memlet_path(nsdfg,
                          exit_node,
                          state.add_write('out'),
                          src_conn='out',
                          memlet=dace.Memlet('out[0:NX, 0:NY, 0:NZ]'))
    sdfg.validate()
    return sdfg


def test_symbolic_kernel_scratch_is_promoted_out_of_the_kernel():
    """The buffer must leave device-local storage and gain one dimension per kernel map parameter."""
    from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import (GPUCodegenPreprocessPipeline)

    sdfg = kernel_with_symbolic_scratch()
    GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})

    descs = [nested.arrays['tmp'] for nested in sdfg.all_sdfgs_recursive() if 'tmp' in nested.arrays]
    assert descs, 'the scratch buffer vanished entirely'
    for desc in descs:
        assert desc.storage == dtypes.StorageType.GPU_Global, desc.storage
        assert len(desc.shape) == 3, desc.shape


@pytest.mark.parametrize('halo', [0, 1])
def test_symbolic_kernel_scratch_indices_stay_inside_the_lifted_buffer(halo):
    """Every kernel iteration must address its own slice INSIDE the lifted buffer.

    The lift sizes each new dimension by the map's trip count -- ``NX - 2`` for the interior sweep
    ``1:NX-1`` -- so an access counts from the map's first iteration. Indexed by the raw map
    parameter, the last iteration wrote one row past the allocation (BOUT++ Hasegawa-Wakatani
    faulted on the GPU). Checked on evaluated flat offsets, not on their spelling.
    """
    from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import (GPUCodegenPreprocessPipeline)

    sdfg = kernel_with_symbolic_scratch(halo)
    GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})

    nx, ny, nz = 6, 5, 3
    sizes = {'NX': nx, 'NY': ny, 'NZ': nz}
    body = next(nested for nested in sdfg.all_sdfgs_recursive() if nested.name == 'scratch_body')
    desc = body.arrays['tmp']
    total = int(dace.symbolic.evaluate(desc.total_size, sizes))
    strides = [int(dace.symbolic.evaluate(s, sizes)) for s in desc.strides]

    def flat_offset(corner, values):
        return sum(int(dace.symbolic.evaluate(bound, values)) * s for bound, s in zip(corner, strides))

    accesses = [e.data.subset for state in body.all_states() for e in state.edges() if e.data.data == 'tmp']
    assert accesses, 'the definition site holds no access to the scratch buffer'
    iterations = [(i, j) for i in range(halo, nx - halo) for j in range(halo, ny - halo)]
    for subset in accesses:
        for i, j in iterations:
            for k in range(nz):
                values = {**sizes, 'i': i, 'j': j, 'k': k}
                lo, hi = flat_offset(subset.min_element(), values), flat_offset(subset.max_element(), values)
                assert 0 <= lo and hi < total, f'{subset} at (i={i}, j={j}, k={k}) reaches {lo}..{hi} of {total}'
        firsts = {flat_offset(subset.min_element(), {**sizes, 'i': i, 'j': j, 'k': 0}) for i, j in iterations}
        assert len(firsts) == len(iterations), f'{subset}: two iterations share one slice'


@pytest.mark.gpu
@pytest.mark.parametrize('halo', [0, 1])
def test_symbolic_kernel_scratch_computes_the_right_values(halo):
    """Structure is not enough here: the shape was already right when the numbers were wrong.

    The stale-body bug reshaped the descriptor and the memlets correctly, compiled cleanly, and
    still had every kernel iteration read and write the same leading slice.
    """
    cupy = pytest.importorskip('cupy')

    nx, ny, nz = 5, 4, 7
    rng = np.random.default_rng(0)
    host_a = rng.random((nx, ny, nz))
    expected = np.zeros((nx, ny, nz))
    interior = (slice(halo, nx - halo), slice(halo, ny - halo))
    expected[interior] = host_a[interior][:, :, ::-1] * 2.0 + 1.0

    out = cupy.zeros((nx, ny, nz))
    kernel_with_symbolic_scratch(halo)(a=cupy.asarray(host_a), out=out, NX=nx, NY=ny, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), expected)


if __name__ == '__main__':
    test_symbolic_kernel_scratch_is_promoted_out_of_the_kernel()
    test_symbolic_kernel_scratch_indices_stay_inside_the_lifted_buffer(1)
    test_symbolic_kernel_scratch_computes_the_right_values(1)
