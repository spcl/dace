# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``compiler.cpu.codegen_params.scalar_emission_type``: normalize single-value TRANSIENTS
    to a by-value ``Scalar`` (``scalar``) or a length-1 ``Array`` (``len1_array``), or leave the mix
    untouched (``keep``, the default). Both conversions are transient-only, so the SDFG signature is
    never rewritten, and ``scalar`` keeps GPU kernel outputs as length-1 arrays. """
import numpy as np
import pytest

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


def descriptors(implementation, mode):
    """``{name: (descriptor_class_name, transient)}`` after the readable pipeline's conversion.

    Replays the two conversion passes the codegen gate runs, so the descriptor effect is observable
    without reaching into ``generate_code``'s internal deep copy.
    """
    from dace.transformation.passes.length_one_array_scalar_conversion import (ConvertLengthOneArraysToScalars,
                                                                               ConvertScalarsToLengthOneArrays)
    sdfg = mixed.to_sdfg(simplify=True)
    if implementation == 'experimental_readable' and mode == 'scalar':
        ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    elif implementation == 'experimental_readable' and mode == 'len1_array':
        ConvertScalarsToLengthOneArrays(transient_only=True).apply_pass(sdfg, {})
    return {name: (type(desc).__name__, desc.transient) for name, desc in sdfg.arrays.items()}


def generate(implementation, mode):
    sdfg = mixed.to_sdfg(simplify=True)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_emission_type', value=mode):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_scalar_makes_single_value_transients_scalars():
    """ ``scalar``: every length-1-array transient becomes a Scalar; signature arrays stay Arrays. """
    kinds = descriptors('experimental_readable', 'scalar')
    assert kinds['buf'] == ('Scalar', True), kinds
    assert kinds['A'][0] == 'Array' and not kinds['A'][1]  # signature input
    assert kinds['out'][0] == 'Array' and not kinds['out'][1]  # signature output -- never rewritten


def test_len1_array_makes_single_value_transients_arrays():
    """ ``len1_array``: every scalar transient becomes a length-1 Array; signature arrays stay Arrays. """
    kinds = descriptors('experimental_readable', 'len1_array')
    assert kinds['s'] == ('Array', True), kinds
    assert kinds['A'][0] == 'Array' and not kinds['A'][1]
    assert kinds['out'][0] == 'Array' and not kinds['out'][1]


def test_keep_is_the_default_and_converts_nothing():
    """ ``keep`` (default) leaves the descriptor mix and the emitted code byte-identical. """
    assert generate('experimental_readable', 'keep') == generate('experimental_readable', None)
    kinds = descriptors('experimental_readable', 'keep')
    # The frontend leaves a genuine mix: a scalar ``s`` and a length-1-array ``buf``.
    assert kinds['s'][0] == 'Scalar' and kinds['buf'][0] == 'Array', kinds


def test_legacy_ignores_the_flag():
    """ Legacy never enters the readable pipeline: byte-identical across every value. """
    assert generate('legacy', 'scalar') == generate('legacy', 'len1_array') == generate('legacy', 'keep')


@pytest.mark.parametrize('mode', ['keep', 'scalar', 'len1_array'])
def test_all_modes_compile_and_run(mode):
    """ Every mode produces the same, correct numbers -- the conversions are semantics-preserving. """
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_emission_type', value=mode):
        A = np.arange(8, dtype=np.float64)
        out = np.zeros(1)
        mixed(A=A.copy(), out=out)
        assert np.isclose(out[0], A.sum() * 2.0)


def test_scalar_keeps_gpu_kernel_output_as_length1_array():
    """ ``scalar`` scalarizes a GPU_Global length-1 output but PromoteGPUScalarsToArrays widens it
    back -- a by-value Scalar cannot live in device memory. Needs no GPU: it is a pure SDFG rewrite. """
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

    ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['acc'], dace.data.Scalar), 'scalarization should first make it a Scalar'
    Pipeline([InferDefaultSchedulesAndStorages(), PromoteGPUScalarsToArrays()]).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['acc'], dace.data.Array), 'GPU kernel output must be widened back to an Array'
    assert tuple(int(s) for s in sdfg.arrays['acc'].shape) == (1, )


if __name__ == '__main__':
    test_scalar_makes_single_value_transients_scalars()
    test_len1_array_makes_single_value_transients_arrays()
    test_keep_is_the_default_and_converts_nothing()
    test_legacy_ignores_the_flag()
    test_all_modes_compile_and_run('keep')
    test_all_modes_compile_and_run('scalar')
    test_all_modes_compile_and_run('len1_array')
    test_scalar_keeps_gpu_kernel_output_as_length1_array()
