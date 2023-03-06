# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Optional
import dace
from dace import nodes
from dace.properties import CodeBlock
import numpy as np
import pytest

from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport


def _instrument(sdfg: dace.SDFG, instr: dace.DataInstrumentationType, ignore: Optional[str] = None):
    # Set instrumentation on all access nodes
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            if ignore and ignore in node.data:
                node.instrument = dace.DataInstrumentationType.No_Instrumentation
            else:
                node.instrument = instr


def test_dump():
    @dace.program
    def tester(A: dace.float64[20, 20]):
        tmp = A + 1
        return tmp + 5

    sdfg = tester.to_sdfg(simplify=True)
    _instrument(sdfg, dace.DataInstrumentationType.Save)

    A = np.random.rand(20, 20)
    result = sdfg(A)
    assert np.allclose(result, A + 6)

    # Verify instrumented data
    dreport = sdfg.get_instrumented_data()
    assert dreport.keys() == {'A', 'tmp', '__return'}
    assert np.allclose(dreport['A'], A)
    assert np.allclose(dreport['tmp'], A + 1)
    assert np.allclose(dreport['__return'], A + 6)


@pytest.mark.gpu
def test_dump_gpu():
    @dace.program
    def tester(A: dace.float64[20, 20]):
        tmp = A + 1
        return tmp + 5

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    _instrument(sdfg, dace.DataInstrumentationType.Save)

    A = np.random.rand(20, 20)
    result = sdfg(A)
    assert np.allclose(result, A + 6)

    # Verify instrumented data
    dreport = sdfg.get_instrumented_data()
    assert dreport.keys() == {'A', 'gpu_A', 'tmp', 'gpu___return', '__return'}
    assert np.allclose(dreport['A'], A)
    assert np.allclose(dreport['gpu_A'], A)
    assert np.allclose(dreport['tmp'], A + 1)
    assert np.allclose(dreport['gpu___return'], A + 6)
    assert np.allclose(dreport['__return'], A + 6)


def test_restore():
    @dace.program
    def tester(A: dace.float64[20, 20]):
        return A + 5

    sdfg = tester.to_sdfg(simplify=True)
    _instrument(sdfg, dace.DataInstrumentationType.Save)

    A = np.random.rand(20, 20)
    acopy = np.copy(A)
    result = sdfg(A)
    assert np.allclose(result, A + 5)

    # Verify instrumented data
    dreport = sdfg.get_instrumented_data()
    _instrument(sdfg, dace.DataInstrumentationType.Restore)

    A[:] = 5
    result = sdfg.call_with_instrumented_data(dreport, A)

    assert np.allclose(result, acopy + 5)


@pytest.mark.gpu
def test_restore_gpu():
    @dace.program
    def tester(A: dace.float64[20, 20]):
        return A + 5

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()

    # Instrument everything but the return value
    _instrument(sdfg, dace.DataInstrumentationType.Save, ignore='return')

    A = np.random.rand(20, 20)
    acopy = np.copy(A)
    result = sdfg(A)
    assert np.allclose(result, A + 5)

    # Verify instrumented data
    dreport = sdfg.get_instrumented_data()
    _instrument(sdfg, dace.DataInstrumentationType.Restore, ignore='return')

    A[:] = 5
    result = sdfg.call_with_instrumented_data(dreport, A)

    assert np.allclose(result, acopy + 5)


def test_dinstr_versioning():
    @dace.program
    def dinstr(A: dace.float64[20], B: dace.float64[20]):
        B[:] = A + 1
        A[:] = B + 1
        B[:] = A + 1

    sdfg = dinstr.to_sdfg(simplify=True)
    _instrument(sdfg, dace.DataInstrumentationType.Save)

    A = np.random.rand(20)
    B = np.random.rand(20)
    oa = np.copy(A)
    sdfg(A, B)

    dreport = sdfg.get_instrumented_data()
    assert len(dreport['A']) == 2
    assert len(dreport['B']) == 2

    assert np.allclose(dreport['A'][0], oa)
    assert np.allclose(dreport['A'][1], oa + 2)
    assert np.allclose(dreport['B'][0], oa + 1)
    assert np.allclose(dreport['B'][1], oa + 3)


def test_dinstr_in_loop():
    @dace.program
    def dinstr(A: dace.float64[20]):
        tmp = np.copy(A)
        for i in range(20):
            tmp[i] = np.sum(tmp)
        return tmp

    sdfg = dinstr.to_sdfg(simplify=True)
    _instrument(sdfg, dace.DataInstrumentationType.Save)

    A = np.random.rand(20)
    result = sdfg(A)
    dreport = sdfg.get_instrumented_data()
    assert len(dreport.keys()) == 3
    assert len(dreport['__return']) == 1 + 2 * 20

    assert np.allclose(dreport['__return'][0], A)
    assert np.allclose(dreport['__return'][-1], result)


def test_dinstr_strided():
    @dace.program
    def dinstr(A: dace.float64[20, 20]):
        tmp = A + 1
        return tmp + 5

    sdfg = dinstr.to_sdfg(simplify=True)
    sdfg.arrays['tmp'].total_size = 32 * 32
    sdfg.arrays['tmp'].strides = (32, 1)

    _instrument(sdfg, dace.DataInstrumentationType.Save, ignore='return')

    A = np.random.rand(20, 20)
    result = sdfg(A)
    assert np.allclose(result, A + 6)

    # Verify instrumented data
    dreport: InstrumentedDataReport = sdfg.get_instrumented_data()
    assert np.allclose(dreport['A'], A)
    assert np.allclose(dreport['tmp'], A + 1)

    # Modify instrumented data and restore
    tmp = dreport['tmp']
    tmp *= 2
    dreport.update_report()

    _instrument(sdfg, dace.DataInstrumentationType.Restore, ignore='return')
    result = sdfg.call_with_instrumented_data(dreport, A=A)
    assert np.allclose(result, 2 * A + 7)


def test_dinstr_symbolic():
    N = dace.symbol('N')

    @dace.program
    def dinstr(A: dace.float64[2 * N, 20]):
        tmp = A + 1
        return tmp + 5

    sdfg = dinstr.to_sdfg(simplify=True)
    _instrument(sdfg, dace.DataInstrumentationType.Save)

    A = np.random.rand(20, 20)
    result = sdfg(A, N=10)
    assert np.allclose(result, A + 6)

    # Verify instrumented data
    dreport: InstrumentedDataReport = sdfg.get_instrumented_data()
    assert np.allclose(dreport['A'], A)
    assert np.allclose(dreport['tmp'], A + 1)


def test_dinstr_in_loop_conditional_cpp():
    @dace.program
    def dinstr(A: dace.float64[20]):
        tmp = np.copy(A)
        for i in range(20):
            tmp[i] = np.sum(tmp)
        return tmp

    sdfg = dinstr.to_sdfg(simplify=True)

    # Set instrumentation on all access nodes
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            node.instrument = dace.DataInstrumentationType.Save
            node.instrument_condition = CodeBlock('i == 0', language=dace.Language.CPP)

    A = np.ones((20,))
    B = np.ones((20,))
    B[0] = 20
    _ = sdfg(A)
    dreport = sdfg.get_instrumented_data()
    assert len(dreport.keys()) == 3
    assert len(dreport['__return']) == 3

    assert np.allclose(dreport['__return'][0], A)
    assert np.allclose(dreport['__return'][-1], B)


def test_dinstr_in_loop_conditional_python():
    @dace.program
    def dinstr(A: dace.float64[20]):
        tmp = np.copy(A)
        for i in range(20):
            tmp[i] = np.sum(tmp)
        return tmp

    sdfg = dinstr.to_sdfg(simplify=True)

    # Set instrumentation on all access nodes
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            node.instrument = dace.DataInstrumentationType.Save
            node.instrument_condition = CodeBlock('i ** 2 == 4', language=dace.Language.Python)

    A = np.ones((20,))
    B = np.ones((20,))
    C = np.ones((20,))
    ret = sdfg(A)
    dreport = sdfg.get_instrumented_data()
    B[0:2] = ret[0:2]
    C[0:3] = ret[0:3]
    assert len(dreport.keys()) == 2
    assert len(dreport['__return']) == 2

    assert np.allclose(dreport['__return'][0], B)
    assert np.allclose(dreport['__return'][1], C)


if __name__ == '__main__':
    test_dump()
    test_dump_gpu()
    test_restore()
    test_restore_gpu()
    test_dinstr_versioning()
    test_dinstr_in_loop()
    test_dinstr_strided()
    test_dinstr_symbolic()
    test_dinstr_in_loop_conditional_cpp()
    test_dinstr_in_loop_conditional_python()
