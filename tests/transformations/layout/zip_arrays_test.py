# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import numpy
import dace

from dace.transformation.layout.zip_arrays import ZipArrays

N = dace.symbol("N")


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])


@dace.program
def mixed(A: dace.float64[N], K: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] + K[i]


def test_zip_homogeneous_fields():
    """Fuse two same-dtype arrays into a field-minor AoS array Z[*S, 2]; run bit-exact."""
    original = madd.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "madd_zipped"
    ZipArrays(zip_map={"Z": ["A", "B"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "A" not in sdfg.arrays and "B" not in sdfg.arrays
    assert "Z" in sdfg.arrays
    assert tuple(str(s) for s in sdfg.arrays["Z"].shape) == ("N", "N", "2")

    _N = 8
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))

    original(A=A.copy(), B=B.copy(), C=C0, N=_N)

    Z = numpy.stack([A, B], axis=-1).copy()  # (N, N, 2) field-minor AoS
    sdfg(Z=Z, C=C1, N=_N)

    assert numpy.allclose(C1, C0)


def test_zip_heterogeneous_struct():
    """Fuse different-dtype arrays into a Structure; access via dotted memlets Z.A[i] / Z.K[i]."""
    sdfg = mixed.to_sdfg()
    sdfg.name = "mixed_zipped"
    ZipArrays(zip_map={"Z": ["A", "K"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "A" not in sdfg.arrays and "K" not in sdfg.arrays
    assert isinstance(sdfg.arrays["Z"], dace.data.Structure)

    _N = 16
    A = numpy.random.rand(_N)
    K = numpy.random.randint(-5, 5, size=_N).astype(numpy.int64)
    C = numpy.zeros(_N)
    ref = A + K.astype(numpy.float64)

    ctype = sdfg.arrays["Z"].dtype._typeclass.as_ctypes()
    z = ctype(A=A.__array_interface__['data'][0], K=K.__array_interface__['data'][0])
    func = sdfg.compile()
    func(Z=z, C=C, N=_N)

    assert numpy.allclose(C, ref)


if __name__ == "__main__":
    test_zip_homogeneous_fields()
    test_zip_heterogeneous_struct()
    print("zip tests PASS")
