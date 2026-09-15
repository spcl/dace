# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for UnzipArrays: the inverse of ZipArrays (homogeneous + struct), a Zip->Unzip roundtrip,
and the nested-SDFG connector-split path."""
import copy
import numpy
import dace

from dace.transformation.layout.zip_arrays import ZipArrays
from dace.transformation.layout.unzip_arrays import UnzipArrays

N = dace.symbol("N")


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])


@dace.program
def mixed(A: dace.float64[N], K: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] + K[i]


def test_zip_unzip_homogeneous_roundtrip():
    """Zip A,B into Z[*,*,2] then Unzip back to A,B: recovers the original, bit-exact."""
    original = madd.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "madd_zip_unzip"
    ZipArrays(zip_map={"Z": ["A", "B"]}).apply_pass(sdfg, {})
    UnzipArrays(unzip_map={"Z": ["A", "B"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "Z" not in sdfg.arrays
    assert "A" in sdfg.arrays and "B" in sdfg.arrays
    assert tuple(str(s) for s in sdfg.arrays["A"].shape) == ("N", "N")

    _N = 8
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))
    original(A=A.copy(), B=B.copy(), C=C0, N=_N)
    sdfg(A=A.copy(), B=B.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


def test_unzip_homogeneous_standalone():
    """Unzip a directly-authored fused array Z[N,2] -> A,B; run sum kernel bit-exact."""
    sdfg = dace.SDFG("unzip_standalone")
    sdfg.add_array("Z", [N, 2], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    state.add_mapped_tasklet(
        name="sum",
        map_ranges={"i": "0:N"},
        inputs={
            "a": dace.Memlet.simple("Z", "i, 0"),
            "b": dace.Memlet.simple("Z", "i, 1")
        },
        code="c = a + b",
        outputs={"c": dace.Memlet.simple("C", "i")},
        external_edges=True,
    )
    UnzipArrays(unzip_map={"Z": ["A", "B"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "Z" not in sdfg.arrays and "A" in sdfg.arrays and "B" in sdfg.arrays

    _N = 10
    A = numpy.random.rand(_N)
    B = numpy.random.rand(_N)
    C = numpy.zeros(_N)
    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, A + B)


def test_zip_unzip_struct_roundtrip():
    """Zip A (f64) + K (i64) into a contiguous array-of-structs then Unzip back: recovers the plain
    arrays, bit-exact."""
    sdfg = mixed.to_sdfg()
    sdfg.name = "mixed_zip_unzip"
    ZipArrays(zip_map={"Z": ["A", "K"]}).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["Z"], dace.data.Array) and isinstance(sdfg.arrays["Z"].dtype, dace.dtypes.struct)
    UnzipArrays(unzip_map={"Z": ["A", "K"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "Z" not in sdfg.arrays
    assert "A" in sdfg.arrays and "K" in sdfg.arrays
    assert sdfg.arrays["K"].dtype == dace.int64

    _N = 12
    A = numpy.random.rand(_N)
    K = numpy.random.randint(-5, 5, size=_N).astype(numpy.int64)
    C = numpy.zeros(_N)
    sdfg(A=A.copy(), K=K.copy(), C=C, N=_N)
    assert numpy.allclose(C, A + K)


def _build_nested_sum_sdfg():
    """Outer SDFG: Z[N,2] -> nested SDFG that indexes Z[i,0]+Z[i,1] -> C[N]."""
    inner = dace.SDFG("inner_sum")
    inner.add_array("Zc", [N, 2], dace.float64)
    inner.add_array("Cc", [N], dace.float64)
    istate = inner.add_state("is", is_start_block=True)
    istate.add_mapped_tasklet(
        name="isum",
        map_ranges={"i": "0:N"},
        inputs={
            "a": dace.Memlet.simple("Zc", "i, 0"),
            "b": dace.Memlet.simple("Zc", "i, 1")
        },
        code="c = a + b",
        outputs={"c": dace.Memlet.simple("Cc", "i")},
        external_edges=True,
    )

    sdfg = dace.SDFG("nested_unzip")
    sdfg.add_array("Z", [N, 2], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {"Zc"}, {"Cc"}, symbol_mapping={"N": N})
    rZ = state.add_read("Z")
    wC = state.add_write("C")
    state.add_edge(rZ, None, nsdfg, "Zc", dace.Memlet.from_array("Z", sdfg.arrays["Z"]))
    state.add_edge(nsdfg, "Cc", wC, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg, nsdfg


def test_unzip_nested_sdfg():
    """A fused array flowing WHOLE into a nested SDFG: the connector splits into 2, inner unzips."""
    sdfg, nsdfg = _build_nested_sum_sdfg()
    UnzipArrays(unzip_map={"Z": ["A", "B"]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert "Z" not in sdfg.arrays and "A" in sdfg.arrays and "B" in sdfg.arrays
    # connector split into two inner field connectors
    assert set(nsdfg.in_connectors) == {"Zc_0", "Zc_1"}
    assert "Zc" not in nsdfg.sdfg.arrays
    assert "Zc_0" in nsdfg.sdfg.arrays and "Zc_1" in nsdfg.sdfg.arrays

    _N = 9
    A = numpy.random.rand(_N)
    B = numpy.random.rand(_N)
    C = numpy.zeros(_N)
    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, A + B)


if __name__ == "__main__":
    test_zip_unzip_homogeneous_roundtrip()
    test_unzip_homogeneous_standalone()
    test_zip_unzip_struct_roundtrip()
    test_unzip_nested_sdfg()
    print("unzip tests PASS")
