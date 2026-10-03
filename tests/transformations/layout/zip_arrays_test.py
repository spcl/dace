# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import numpy
import dace

from dace.transformation.layout.zip_arrays import ZipArrays
from dace.transformation.layout.unzip_arrays import UnzipArrays

N = dace.symbol("N")
KRED = 5


def _wcr_count(sdfg):
    return sum(1 for st in sdfg.all_states() for e in st.edges() if e.data is not None and e.data.wcr is not None)


@dace.program
def dual_reduce(x: dace.float64[N], acc0: dace.float64[N], acc1: dace.float64[N]):
    for i, j in dace.map[0:N, 0:KRED]:
        acc0[i] += x[i]
        acc1[i] += x[i] * 2.0


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


def test_zip_preserves_wcr_reduction():
    """Zipping two homogeneous WCR reduction targets (acc0/acc1) must keep their wcr, else the
    ``+=`` degrades to last-writer-wins. The fused Z[i, k] accumulates bit-exactly."""
    _N = 8
    x = numpy.random.default_rng(0).random(_N)
    ref0, ref1 = KRED * x, 2.0 * KRED * x

    sdfg = dual_reduce.to_sdfg(simplify=True)
    assert _wcr_count(sdfg) == 2
    ZipArrays(zip_map={"Z": ["acc0", "acc1"]}).apply_pass(sdfg, {})
    assert _wcr_count(sdfg) == 2, "Zip dropped the WCR"
    sdfg.validate()

    Z = numpy.zeros((_N, 2))
    sdfg(x=x.copy(), Z=Z, N=_N)
    assert numpy.allclose(Z[:, 0], ref0) and numpy.allclose(Z[:, 1], ref1)


def test_zip_unzip_wcr_roundtrip():
    """Zip then Unzip two WCR reduction targets: the wcr survives both memlet rebuilds and the
    reductions stay exact."""
    _N = 8
    x = numpy.random.default_rng(1).random(_N)
    ref0, ref1 = KRED * x, 2.0 * KRED * x

    sdfg = dual_reduce.to_sdfg(simplify=True)
    ZipArrays(zip_map={"Z": ["acc0", "acc1"]}).apply_pass(sdfg, {})
    UnzipArrays(unzip_map={"Z": ["acc0", "acc1"]}).apply_pass(sdfg, {})
    assert _wcr_count(sdfg) == 2, "Zip->Unzip dropped the WCR"
    sdfg.validate()

    a0, a1 = numpy.zeros(_N), numpy.zeros(_N)
    sdfg(x=x.copy(), acc0=a0, acc1=a1, N=_N)
    assert numpy.allclose(a0, ref0) and numpy.allclose(a1, ref1)


if __name__ == "__main__":
    test_zip_homogeneous_fields()
    test_zip_heterogeneous_struct_true_aos()
    test_zip_heterogeneous_struct_padded_dtypes()
    test_zip_preserves_wcr_reduction()
    test_zip_unzip_wcr_roundtrip()
    print("zip tests PASS")
