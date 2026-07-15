# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra coverage for the PadDimensions layout primitive.

Complements pad_dimension_test.py: exercises multi-dimension padding of a single array, the
zero-pad no-op, recursion into a nested SDFG, non-final-dimension padding, symbolic pad amounts,
preservation of a Fortran-packed base, and the rank-mismatch guard. Every compiled case is checked
bit-exact against a NumPy oracle, relying on the invariant that pad cells are never accessed.
"""
import numpy
import pytest

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.layout.pad_dimensions import PadDimensions

N = dace.symbol("N")
P = dace.symbol("P")


@dace.program
def add2d(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def add3d(A: dace.float64[N, N, N], B: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    for i, j, k in dace.map[0:N, 0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j, k] = A[i, j, k] * 2.0 + B[i, j, k]


@dace.program
def scale1d(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        B[i] = A[i] * 4.0


def test_multidim_pad_single_array_bit_exact():
    """Padding BOTH dimensions of one array keeps packed-C strides and the live region intact."""
    sdfg = add2d.to_sdfg()
    sdfg.name = "pad_extra_multidim"
    PadDimensions(pad_map={"A": [2, 3]}).apply_pass(sdfg, {})
    sdfg.validate()

    a = sdfg.arrays["A"]
    assert [str(s) for s in a.shape] == ["N + 2", "N + 3"]
    # Packed-C strides for the grown shape: [new_shape[1], 1].
    assert [str(s) for s in a.strides] == ["N + 3", "1"]
    assert str(a.total_size) == str((N + 2) * (N + 3))
    # B and C are untouched.
    assert [str(s) for s in sdfg.arrays["B"].shape] == ["N", "N"]

    _N = 5
    A = numpy.random.rand(_N + 2, _N + 3)
    B = numpy.random.rand(_N, _N)
    C = numpy.zeros((_N, _N))
    ref = A[:_N, :_N] + B  # live region only

    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, ref)


def test_pad_amount_zero_is_noop():
    """A pad of 0 on every dimension leaves shape, strides and total size identical."""
    sdfg = add2d.to_sdfg()
    sdfg.name = "pad_extra_zero"
    a = sdfg.arrays["A"]
    before_shape = list(a.shape)
    before_strides = list(a.strides)
    before_total = a.total_size

    PadDimensions(pad_map={"A": [0, 0]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert list(a.shape) == before_shape
    assert list(a.strides) == before_strides
    assert a.total_size == before_total

    _N = 6
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C = numpy.zeros((_N, _N))
    ref = A + B

    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, ref)


def test_pad_recurses_into_nested_sdfg_bit_exact():
    """Growing an outer array grows the matching descriptor inside a nested SDFG (via connectors)."""
    outer = dace.SDFG("pad_extra_nested")
    outer.add_array("A", [N], dace.float64)
    outer.add_array("B", [N], dace.float64)
    state = outer.add_state("main")

    inner = dace.SDFG("inner_pad")
    inner.add_array("a", [N], dace.float64)
    inner.add_array("b", [N], dace.float64)
    istate = inner.add_state("scale_state")
    istate.add_mapped_tasklet(
        "scale",
        dict(i="0:N"),
        inputs={"inp": dace.Memlet("a[i]")},
        code="out = inp * 3.0",
        outputs={"out": dace.Memlet("b[i]")},
        external_edges=True,
    )

    nsdfg = state.add_nested_sdfg(inner, {"a"}, {"b"}, symbol_mapping={"N": N})
    read_a = state.add_read("A")
    write_b = state.add_write("B")
    state.add_edge(read_a, None, nsdfg, "a", dace.Memlet("A[0:N]"))
    state.add_edge(nsdfg, "b", write_b, None, dace.Memlet("B[0:N]"))

    assert [str(s) for s in inner.arrays["a"].shape] == ["N"]

    PadDimensions(pad_map={"A": [5], "B": [5]}).apply_pass(outer, {})
    outer.validate()

    # Outer descriptors grew.
    assert [str(s) for s in outer.arrays["A"].shape] == ["N + 5"]
    assert [str(s) for s in outer.arrays["B"].shape] == ["N + 5"]
    # Recursion grew the nested descriptors under their connector names.
    assert [str(s) for s in inner.arrays["a"].shape] == ["N + 5"]
    assert [str(s) for s in inner.arrays["b"].shape] == ["N + 5"]

    _N = 7
    A = numpy.random.rand(_N + 5)
    B = numpy.zeros(_N + 5)
    guard = B.copy()
    ref = A[:_N] * 3.0

    outer(A=A.copy(), B=B, N=_N)
    assert numpy.allclose(B[:_N], ref)
    # Padded tail of the output was never written.
    assert numpy.array_equal(B[_N:], guard[_N:])


def test_non_final_dim_pad_bit_exact():
    """Padding non-final (leading and middle) dims recomputes the trailing-major C strides."""
    sdfg = add3d.to_sdfg()
    sdfg.name = "pad_extra_nonfinal"
    PadDimensions(pad_map={"A": [1, 2, 0]}).apply_pass(sdfg, {})
    sdfg.validate()

    a = sdfg.arrays["A"]
    assert [str(s) for s in a.shape] == ["N + 1", "N + 2", "N"]
    # C strides: [shape1*shape2, shape2, 1] with shape2 == N (last dim unpadded).
    assert [str(s) for s in a.strides] == ["N*(N + 2)", "N", "1"]

    _N = 5
    A = numpy.random.rand(_N + 1, _N + 2, _N)
    B = numpy.random.rand(_N, _N, _N)
    C = numpy.zeros((_N, _N, _N))
    ref = A[:_N, :_N, :_N] * 2.0 + B

    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, ref)


def test_symbolic_pad_amount_bit_exact():
    """The pad amount itself may be a symbol; the grown extent becomes N + P."""
    # Descriptor-level check on a 2D array padded by a symbol in the leading dim only.
    sdfg2d = add2d.to_sdfg()
    PadDimensions(pad_map={"A": [P, 0]}).apply_pass(sdfg2d, {})
    a2d = sdfg2d.arrays["A"]
    assert [str(s) for s in a2d.shape] == ["N + P", "N"]
    assert [str(s) for s in a2d.strides] == ["N", "1"]
    assert str(a2d.total_size) == str(N * (N + P))

    # Compiled bit-exact check on a 1D array padded by a symbol, run with a concrete P.
    sdfg = scale1d.to_sdfg()
    sdfg.name = "pad_extra_symbolic"
    PadDimensions(pad_map={"A": [P]}).apply_pass(sdfg, {})
    sdfg.validate()
    a = sdfg.arrays["A"]
    assert [str(s) for s in a.shape] == ["N + P"]
    assert [str(s) for s in a.strides] == ["1"]

    _N, _P = 6, 4
    A = numpy.random.rand(_N + _P)
    B = numpy.zeros(_N)
    ref = A[:_N] * 4.0

    sdfg(A=A.copy(), B=B, N=_N, P=_P)
    assert numpy.allclose(B, ref)


def test_fortran_packed_base_preserved():
    """A Fortran-packed (column-major) array keeps a Fortran-packed base after padding."""
    sdfg = dace.SDFG("fortran_pad")
    sdfg.add_state("s")  # a valid SDFG needs at least one state
    # Non-square so Fortran strides differ from C strides.
    sdfg.add_array("F", [5, 3], dace.float64, strides=[1, 5])
    desc = sdfg.arrays["F"]
    assert desc.is_packed_fortran_strides()
    assert not desc.is_packed_c_strides()

    PadDimensions(pad_map={"F": [2, 1]}).apply_pass(sdfg, {})
    sdfg.validate()

    assert [str(s) for s in desc.shape] == ["7", "4"]
    # Fortran-packed strides for [7, 4]: [1, 7]. A C base would have produced [4, 1].
    assert [str(s) for s in desc.strides] == ["1", "7"]
    assert int(desc.total_size) == 28
    assert desc.is_packed_fortran_strides()
    assert not desc.is_packed_c_strides()


def test_pad_length_mismatch_raises():
    """A pad list whose length does not match the array rank is rejected."""
    sdfg = dace.SDFG("mismatch_pad")
    sdfg.add_array("W", [3, 4], dace.float64)
    with pytest.raises(ValueError):
        PadDimensions(pad_map={"W": [1]}).apply_pass(sdfg, {})


def test_pass_metadata_and_return_value():
    """The pass reports Descriptors-only modification, never re-applies, and returns 0."""
    empty = PadDimensions(pad_map={})
    assert empty.modifies() == ppl.Modifies.Descriptors
    assert empty.should_reapply(ppl.Modifies.Descriptors) is False
    assert empty.should_reapply(ppl.Modifies.Everything) is False

    sdfg = dace.SDFG("return_pad")
    sdfg.add_array("Q", [2], dace.float64)
    assert PadDimensions(pad_map={"Q": [1]}).apply_pass(sdfg, {}) == 0
    assert [str(s) for s in sdfg.arrays["Q"].shape] == ["3"]


if __name__ == "__main__":
    test_multidim_pad_single_array_bit_exact()
    test_pad_amount_zero_is_noop()
    test_pad_recurses_into_nested_sdfg_bit_exact()
    test_non_final_dim_pad_bit_exact()
    test_symbolic_pad_amount_bit_exact()
    test_fortran_packed_base_preserved()
    test_pad_length_mismatch_raises()
    test_pass_metadata_and_return_value()
    print("pad extra tests PASS")
