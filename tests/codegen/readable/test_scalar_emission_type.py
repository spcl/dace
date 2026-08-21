# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the readable generator's scalarization of single-value transients."""
import numpy as np

import dace
from dace.config import set_temporary


@dace.program
def mixed(A: dace.float64[8], out: dace.float64[1]):
    # ``out`` is a NON-transient length-1 array -- part of the signature, must never be rewritten.
    s = np.float64(0.0)  # scalar transient
    buf = np.zeros((1, ), np.float64)  # length-1-array transient
    for i in dace.map[0:8]:
        s = s + A[i]
    buf[0] = s * 2.0
    out[0] = buf[0]


def test_single_value_transients_become_scalars():
    """Every length-1-array transient becomes a Scalar; signature arrays stay Arrays."""
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    sdfg = mixed.to_sdfg(simplify=True)
    ConvertLengthOneArraysToScalars(skip_gpu_outputs=True).apply_pass(sdfg, {})
    kinds = {name: (type(desc).__name__, desc.transient) for name, desc in sdfg.arrays.items()}
    assert kinds['buf'] == ('Scalar', True), kinds
    assert kinds['s'] == ('Scalar', True), kinds
    assert kinds['A'][0] == 'Array' and not kinds['A'][1]  # signature input
    assert kinds['out'][0] == 'Array' and not kinds['out'][1]  # signature output -- never rewritten


def test_readable_compiles_and_runs():
    """Scalarization is semantics-preserving."""
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        A = np.arange(8, dtype=np.float64)
        out = np.zeros(1)
        mixed(A=A.copy(), out=out)
        assert np.isclose(out[0], A.sum() * 2.0)


def test_scalar_keeps_gpu_kernel_output_as_length1_array():
    """A GPU_Global output is scalarized and then widened back: a by-value Scalar cannot live in
    device memory."""
    from dace.transformation.pass_pipeline import Pipeline
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    from dace.transformation.passes.promote_gpu_scalars_to_arrays import (InferDefaultSchedulesAndStorages,
                                                                          PromoteGPUScalarsToArrays)
    sdfg = dace.SDFG('gpu_out')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('acc', [1], dace.float64, transient=True, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state('s')
    state.add_mapped_tasklet('k',
                             dict(i='0:8'), {'a': dace.Memlet('A[i]')},
                             'o = a',
                             dict(o=dace.Memlet('acc[0]', wcr='lambda x, y: x + y')),
                             schedule=dace.ScheduleType.GPU_Device,
                             external_edges=True)

    ConvertLengthOneArraysToScalars(skip_gpu_outputs=True).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['acc'], dace.data.Scalar), 'scalarization should first make it a Scalar'
    Pipeline([InferDefaultSchedulesAndStorages(), PromoteGPUScalarsToArrays()]).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['acc'], dace.data.Array), 'GPU kernel output must be widened back to an Array'
    assert tuple(int(s) for s in sdfg.arrays['acc'].shape) == (1, )


if __name__ == '__main__':
    test_single_value_transients_become_scalars()
    test_readable_compiles_and_runs()
    test_scalar_keeps_gpu_kernel_output_as_length1_array()
