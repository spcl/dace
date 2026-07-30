# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Low-precision dtypes bfloat16 / float8_e4m3fn / float8_e5m2 (ml_dtypes-backed).

Covers registration, numpy interop (passing ml_dtypes arrays in and out),
descriptor + SDFG serialization (including symbolic shapes), CPU compilation and
bit-exact execution, and -- on a GPU -- that the CPU struct and the CUDA native
type share one byte representation so a host<->device copy is well-defined.
"""
import ml_dtypes
import numpy as np
import pytest

import dace
from dace import dtypes

# (typeclass, ml_dtypes scalar, byte width, ctype string, raw-view numpy dtype)
LOWP = [
    (dace.bfloat16, ml_dtypes.bfloat16, 2, "dace::bfloat16", np.uint16),
    (dace.float8_e4m3fn, ml_dtypes.float8_e4m3fn, 1, "dace::float8_e4m3fn", np.uint8),
    (dace.float8_e5m2, ml_dtypes.float8_e5m2, 1, "dace::float8_e5m2", np.uint8),
]

# A spread that hits normals, subnormals, sign, and the overflow edges of each format.
SAMPLE = np.array([0.0, 1.0, -1.0, 0.5, -2.0, 3.14159, 0.001, 255.0, 448.0, 480.0, 57344.0, -1e9], dtype=np.float32)


@pytest.mark.parametrize("tc,scalar,nbytes,ctype,raw", LOWP)
def test_typeclass_registered(tc, scalar, nbytes, ctype, raw):
    assert tc.to_string() == scalar.__name__  # named verbatim as ml_dtypes names it
    assert tc.type is scalar
    assert tc.bytes == nbytes
    assert tc.ctype == ctype
    assert tc.as_numpy_dtype() == np.dtype(scalar)
    assert tc in dtypes.FLOAT_TYPES
    assert scalar.__name__ in dtypes.TYPECLASS_STRINGS
    assert dtypes.dtype_to_typeclass(scalar) is tc


@pytest.mark.parametrize("tc,scalar,nbytes,ctype,raw", LOWP)
def test_descriptor_serialization_roundtrip(tc, scalar, nbytes, ctype, raw):
    # Symbolic shape, so the symbolic-property path is exercised too.
    N = dace.symbol('N')
    desc = dace.data.Array(tc, [N, 4])
    desc2 = dace.serialize.from_json(desc.to_json(), {"version": dace.__version__})
    assert desc2.dtype == tc
    assert str(desc2.shape) == str(desc.shape)


def test_sdfg_serialization_roundtrip():
    # Full SDFG (JSON + .sdfgz) with symbolic-shaped low-precision arrays.
    N = dace.symbol('N')
    sdfg = dace.SDFG('lowp')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('a', [N], dace.bfloat16)
    sdfg.add_array('b', [N], dace.float8_e4m3fn)
    sdfg.add_array('c', [N], dace.float8_e5m2)

    sd2 = dace.SDFG.from_json(sdfg.to_json())
    assert (sd2.arrays['a'].dtype, sd2.arrays['b'].dtype, sd2.arrays['c'].dtype) == \
           (dace.bfloat16, dace.float8_e4m3fn, dace.float8_e5m2)

    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), 'lowp_roundtrip.sdfgz')
    sdfg.save(path)
    assert dace.SDFG.from_file(path).arrays['a'].dtype == dace.bfloat16


def _run_add_one(tc, arr):
    """Compile and run ``b = a + 1`` for one low-precision dtype; returns b."""
    if tc is dace.bfloat16:

        @dace.program
        def k(a: dace.bfloat16[arr.size], b: dace.bfloat16[arr.size]):
            b[:] = a + 1
    elif tc is dace.float8_e4m3fn:

        @dace.program
        def k(a: dace.float8_e4m3fn[arr.size], b: dace.float8_e4m3fn[arr.size]):
            b[:] = a + 1
    else:

        @dace.program
        def k(a: dace.float8_e5m2[arr.size], b: dace.float8_e5m2[arr.size]):
            b[:] = a + 1

    out = np.zeros_like(arr)
    k(arr, out)
    return out


@pytest.mark.parametrize("tc,scalar,nbytes,ctype,raw", LOWP)
def test_numpy_roundtrip_and_compute(tc, scalar, nbytes, ctype, raw):
    # Pass an ml_dtypes array in from outside, compute, get it back -- bit-exact vs ml_dtypes.
    a = SAMPLE.astype(scalar)
    out = _run_add_one(tc, a)
    expected = (a.astype(np.float32) + 1).astype(scalar)
    np.testing.assert_array_equal(out.view(raw), expected.view(raw))


@pytest.mark.gpu
@pytest.mark.parametrize("tc,scalar,nbytes,ctype,raw", LOWP)
def test_gpu_cpu_representation_equivalence(tc, scalar, nbytes, ctype, raw):
    # Identity host->device->host copy. The GPU array is the CUDA native type; if it did not share
    # the CPU struct's byte layout, the round trip would corrupt the bytes or crash. This is the copy
    # safety the whole feature exists for. (fp8 has no device ALU, so an identity copy -- not
    # arithmetic -- is the right probe for all three formats.)
    if tc is dace.bfloat16:

        @dace.program
        def kid(a: dace.bfloat16[SAMPLE.size], b: dace.bfloat16[SAMPLE.size]):
            b[:] = a
    elif tc is dace.float8_e4m3fn:

        @dace.program
        def kid(a: dace.float8_e4m3fn[SAMPLE.size], b: dace.float8_e4m3fn[SAMPLE.size]):
            b[:] = a
    else:

        @dace.program
        def kid(a: dace.float8_e5m2[SAMPLE.size], b: dace.float8_e5m2[SAMPLE.size]):
            b[:] = a

    a = SAMPLE.astype(scalar)
    sdfg = kid.to_sdfg()
    sdfg.apply_gpu_transformations()
    gpu = np.zeros_like(a)
    sdfg(a=a.copy(), b=gpu)
    # Bytes survive the copy boundary unchanged == CPU struct and CUDA native type share one layout.
    np.testing.assert_array_equal(gpu.view(raw), a.view(raw))


def test_openmp_reduction_bf16():
    # Forces the OpenMP Reduce expansion, which emits `reduction(+: out)` over dace::bfloat16 and so
    # needs the `declare reduction` the runtime header provides. N <= 256 keeps a bf16 sum of ones
    # exact under any thread split, so the result is asserted with zero tolerance.
    from dace.libraries.standard.nodes.reduce import Reduce
    N = 128

    @dace.program
    def red(a: dace.bfloat16[N], r: dace.bfloat16[1]):
        r[0] = np.sum(a)

    sdfg = red.to_sdfg(simplify=True)
    reduces = [n for state in sdfg.states() for n in state.nodes() if isinstance(n, Reduce)]
    assert reduces, "expected a Reduce node to force the OpenMP expansion"
    for n in reduces:
        n.implementation = 'OpenMP'

    a = np.ones(N, dtype=ml_dtypes.bfloat16)
    r = np.zeros(1, dtype=ml_dtypes.bfloat16)
    sdfg(a=a, r=r)
    assert float(r[0]) == float(N)


@pytest.mark.gpu
def test_gpu_cpu_compute_equivalence_bf16():
    # bfloat16 has device arithmetic (__nv_bfloat16), so CPU and GPU must produce identical bytes.
    a = SAMPLE.astype(ml_dtypes.bfloat16)
    cpu = _run_add_one(dace.bfloat16, a)

    @dace.program
    def kg(a: dace.bfloat16[a.size], b: dace.bfloat16[a.size]):
        b[:] = a + 1

    sdfg = kg.to_sdfg()
    sdfg.apply_gpu_transformations()
    gpu = np.zeros_like(a)
    sdfg(a=a.copy(), b=gpu)
    np.testing.assert_array_equal(gpu.view(np.uint16), cpu.view(np.uint16))


if __name__ == "__main__":
    for args in LOWP:
        test_typeclass_registered(*args)
        test_descriptor_serialization_roundtrip(*args)
        test_numpy_roundtrip_and_compute(*args)
    test_sdfg_serialization_roundtrip()
    test_openmp_reduction_bf16()
    print("fp8/bf16 dtype tests passed")
