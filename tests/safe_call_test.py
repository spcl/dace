# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

skip_sdfg_safe_call_on_nanobind = pytest.mark.skipif(
    dace.Config.get('compiler', 'interface') == 'nanobind',
    reason='SDFG.safe_call() is refused on nanobind by design: it hides the compiled object, whose '
    'collision rename would make post-call queries on the original SDFG unsound; '
    'use compile() + CompiledSDFG.safe_call() (the *_precompiled variants).')


@dace.program
def write_to_null(A: dace.float64[5], B: dace.float64[5], ub: dace.int64):
    for i in range(5):
        with dace.tasklet("CPP"):
            b << B[i]
            u << ub
            a >> A[i]
            """
                if (u == 0){
                    void* ptr = nullptr;
                    *((double*)ptr) = 42.0;
                }
                a = b + 1;
                """


@pytest.mark.sequential
@skip_sdfg_safe_call_on_nanobind
def test_wtn():
    sdfg = write_to_null.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A, B, 5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"

    # This should raise an exception, but not crash
    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    caught = False
    try:
        sdfg.safe_call(A, B, 0)
        caught = False
    except Exception as e:
        caught = True
    assert caught, "Exception not raised!"


@pytest.mark.sequential
def test_wtn_precompiled():
    sdfg = write_to_null.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    obj = sdfg.compile()
    obj.safe_call(A, B, 5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"

    # This should raise an exception, but not crash
    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    caught = False
    try:
        obj.safe_call(A, B, 0)
        caught = False
    except Exception as e:
        caught = True
    assert caught, "Exception not raised!"


@pytest.mark.sequential
@skip_sdfg_safe_call_on_nanobind
def test_instrumentation():
    sdfg = write_to_null.to_sdfg()
    sdfg.instrument = dace.InstrumentationType.Timer

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A, B, 5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"

    assert sdfg.get_latest_report() is not None, "Report not generated!"


@pytest.mark.sequential
def test_instrumentation_precompiled():
    sdfg = write_to_null.to_sdfg()
    sdfg.instrument = dace.InstrumentationType.Timer

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    obj = sdfg.compile()
    obj.safe_call(A, B, 5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"

    # Query the report via the compiled object's sdfg: that is the artifact
    # that actually ran. On the nanobind interface a collision rename may
    # relocate the compile into its own build folder, which the original
    # `sdfg` object does not know about; `obj.sdfg` always does.
    assert obj.sdfg.get_latest_report() is not None, "Report not generated!"


@pytest.mark.sequential
@skip_sdfg_safe_call_on_nanobind
def test_kwargs():
    sdfg = write_to_null.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A=A, B=B, ub=5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"


@pytest.mark.sequential
def test_kwargs_precompiled():
    sdfg = write_to_null.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    obj = sdfg.compile()
    obj.safe_call(A=A, B=B, ub=5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"


@pytest.mark.sequential
@skip_sdfg_safe_call_on_nanobind
def test_symbols():
    N = dace.symbol('N')

    @dace.program
    def indirect_access_sym(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            A[i] = B[i] + 1

    sdfg = indirect_access_sym.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A=A, B=B, N=5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"


@pytest.mark.sequential
def test_symbols_precompiled():
    N = dace.symbol('N')

    @dace.program
    def indirect_access_sym(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            A[i] = B[i] + 1

    sdfg = indirect_access_sym.to_sdfg()
    obj = sdfg.compile()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    obj.safe_call(A=A, B=B, N=5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"


if __name__ == "__main__":
    test_wtn()
    test_wtn_precompiled()
    test_instrumentation()
    test_instrumentation_precompiled()
    test_kwargs()
    test_kwargs_precompiled()
    test_symbols()
    test_symbols_precompiled()
