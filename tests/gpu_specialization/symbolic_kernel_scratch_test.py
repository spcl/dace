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


def kernel_with_symbolic_scratch() -> dace.SDFG:
    """``out[i, j, k] = 2 * a[i, j, NZ - 1 - k] + 1`` through a per-iteration ``tmp[NZ]``.

    The reversed read is what keeps ``tmp`` alive: a straight-through copy is recomputed into the
    consumer and the buffer disappears.
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

    sdfg = dace.SDFG('kernel_with_symbolic_scratch')
    sdfg.add_array('a', [NX, NY, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [NX, NY, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('body', is_start_block=True)
    entry, exit_node = state.add_map('grid', dict(i='0:NX', j='0:NY'), schedule=dtypes.ScheduleType.GPU_Device)
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


@pytest.mark.gpu
def test_symbolic_kernel_scratch_computes_the_right_values():
    """Structure is not enough here: the shape was already right when the numbers were wrong.

    The stale-body bug reshaped the descriptor and the memlets correctly, compiled cleanly, and
    still had every kernel iteration read and write the same leading slice.
    """
    cupy = pytest.importorskip('cupy')

    nx, ny, nz = 5, 4, 7
    rng = np.random.default_rng(0)
    host_a = rng.random((nx, ny, nz))
    expected = host_a[:, :, ::-1] * 2.0 + 1.0

    out = cupy.zeros((nx, ny, nz))
    kernel_with_symbolic_scratch()(a=cupy.asarray(host_a), out=out, NX=nx, NY=ny, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), expected)


if __name__ == '__main__':
    test_symbolic_kernel_scratch_is_promoted_out_of_the_kernel()
    test_symbolic_kernel_scratch_computes_the_right_values()
