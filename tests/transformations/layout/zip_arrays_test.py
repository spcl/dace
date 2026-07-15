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


@dace.program
def mixed3(A: dace.float64[N], K: dace.int32[N], F: dace.float32[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] + K[i] + F[i]


def test_zip_heterogeneous_struct_true_aos():
    """Fuse different-dtype arrays into ONE contiguous array of structs (true interleaved AoS):
    Z is Array(dtype=struct), a field access A[i] becomes a whole-struct memlet Z[i] and the tasklet
    code member-accesses Z.A / Z.K. Emits `struct { double A; int64 K; } Z[N]` (not a struct of
    pointers), so a numpy structured array marshals directly. Runs bit-exact."""
    sdfg = mixed.to_sdfg()
    sdfg.name = "mixed_zipped"
    ZipArrays(zip_map={"Z": ["A", "K"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "A" not in sdfg.arrays and "K" not in sdfg.arrays
    z_desc = sdfg.arrays["Z"]
    assert isinstance(z_desc, dace.data.Array)  # a contiguous ARRAY of structs, not a Structure
    assert isinstance(z_desc.dtype, dace.dtypes.struct)
    # the emitted C is a real struct with VALUE members (no pointers)
    emitted = "\n".join(p.clean_code for p in sdfg.generate_code())
    assert "double* A" not in emitted and "double *A" not in emitted

    _N = 16
    dt = numpy.dtype({"names": ["A", "K"], "formats": [numpy.float64, numpy.int64], "align": True})
    z = numpy.zeros(_N, dtype=dt)
    z["A"] = numpy.random.rand(_N)
    z["K"] = numpy.random.randint(-5, 5, size=_N)
    C = numpy.zeros(_N)
    ref = z["A"] + z["K"].astype(numpy.float64)

    sdfg(Z=z, C=C, N=_N)
    assert numpy.allclose(C, ref)


def test_zip_heterogeneous_struct_padded_dtypes():
    """Three fields of DIFFERENT sizes (f64/i32/f32) -> a C-aligned struct; the numpy structured
    array (align=True) must marshal against dace's struct layout. Runs bit-exact."""
    sdfg = mixed3.to_sdfg()
    sdfg.name = "mixed3_zipped"
    ZipArrays(zip_map={"Z": ["A", "K", "F"]}).apply_pass(sdfg, {})
    sdfg.validate()

    _N = 12
    dt = numpy.dtype({"names": ["A", "K", "F"], "formats": [numpy.float64, numpy.int32, numpy.float32], "align": True})
    z = numpy.zeros(_N, dtype=dt)
    z["A"] = numpy.random.rand(_N)
    z["K"] = numpy.random.randint(-5, 5, size=_N)
    z["F"] = numpy.random.rand(_N).astype(numpy.float32)
    C = numpy.zeros(_N)
    ref = z["A"] + z["K"].astype(numpy.float64) + z["F"].astype(numpy.float64)

    sdfg(Z=z, C=C, N=_N)
    assert numpy.allclose(C, ref)


if __name__ == "__main__":
    test_zip_homogeneous_fields()
    test_zip_heterogeneous_struct_true_aos()
    test_zip_heterogeneous_struct_padded_dtypes()
    print("zip tests PASS")
