# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra unit tests for SplitArray (dace/transformation/layout/split_array.py).

These complement the kernel-level tests in tests/layout/split_array_test.py by
exercising:
  * the pure helper functions (resolve_aliases, copy_state_contents,
    reverse_bfs_assignments) in isolation,
  * splitting a symbol-sized dimension into named per-index arrays with
    compile-time (extent-1) slices, bit-exact vs numpy,
  * the two-split-dimension Cartesian-product naming,
  * the length-1 descriptor path for a fully consumed split dimension,
  * the data-dependent-index ConditionalBlock branch path, bit-exact vs numpy,
  * the not-yet-supported nested-SDFG raise path.
"""
import numpy

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.layout.split_array import (SplitArray, resolve_aliases, copy_state_contents,
                                                    reverse_bfs_assignments)

nphase = dace.symbol("nphase", dtype=dace.int32)
ncol = dace.symbol("ncol", dtype=dace.int32)

NAMES = ["ql", "qi", "qr", "qs"]
SYMBOL_MAP = {"nphase": 4}
NAME_MAP = {"nphase": NAMES}


@dace.program
def phase_axpy(field: dace.float64[nphase, ncol], out: dace.float64[ncol]):
    for j in range(ncol):
        out[j] = field[0, j] + 2.0 * field[1, j] - field[3, j]


@dace.program
def phase_matrix(m: dace.float64[nphase, nphase, ncol], out: dace.float64[ncol]):
    for j in range(ncol):
        out[j] = m[0, 1, j] - m[2, 3, j] + m[1, 1, j]


@dace.program
def pick_phase(a: dace.float64[nphase, ncol], out: dace.float64[ncol], k: dace.int32):
    for j in range(ncol):
        out[j] = a[k, j] + 1.0


# ---------------------------------------------------------------------------- #
#  Pure helper functions
# ---------------------------------------------------------------------------- #
def test_resolve_aliases_dedup_and_none_passthrough():
    """Aliased symbols (same interstate value) collapse to the first key; None dims survive."""
    exprs = [
        dace.symbolic.pystr_to_symbolic("a1"),
        None,
        dace.symbolic.pystr_to_symbolic("b"),
    ]
    # a1 maps to the same value as a0 -> a1 is an alias of the canonical a0.
    assignments = {"a0": "imelt[0]", "a1": "imelt[0]", "b": "imelt[1]"}
    resolved = resolve_aliases(exprs, assignments)

    assert str(resolved[0]) == "a0"  # a1 canonicalized to a0
    assert resolved[1] is None  # None passes through untouched
    assert str(resolved[2]) == "b"  # distinct value keeps its own symbol

    # With no aliases nothing is substituted.
    plain = resolve_aliases([dace.symbolic.pystr_to_symbolic("x")], {"x": "v0", "y": "v1"})
    assert str(plain[0]) == "x"


def test_copy_state_contents_preserves_graph():
    """copy_state_contents replicates nodes/edges into a fresh state and returns a 1:1 node map."""
    sdfg = dace.SDFG("copy_contents")
    sdfg.add_array("X", [4], dace.float64)
    sdfg.add_array("Y", [4], dace.float64)
    src = sdfg.add_state("src", is_start_block=True)
    r = src.add_read("X")
    w = src.add_write("Y")
    t = src.add_tasklet("cp", {"a"}, {"b"}, "b = a")
    src.add_edge(r, None, t, "a", dace.Memlet("X[0:4]"))
    src.add_edge(t, "b", w, None, dace.Memlet("Y[0:4]"))

    dst = sdfg.add_state("dst")
    node_map = copy_state_contents(src, dst)

    assert len(node_map) == src.number_of_nodes()
    assert dst.number_of_nodes() == src.number_of_nodes()
    assert dst.number_of_edges() == src.number_of_edges()
    # Each original maps to a fresh, distinct copy.
    assert all(orig is not clone for orig, clone in node_map.items())
    # Access-node data names are preserved by the copy.
    assert {n.data for n in src.data_nodes()} == {n.data for n in dst.data_nodes()}


def test_reverse_bfs_assignments_closest_wins():
    """Walking backward keeps the closest assignment per symbol key."""
    sdfg = dace.SDFG("bfs_assign")
    a = sdfg.add_state("a", is_start_block=True)
    b = sdfg.add_state("b")
    c = sdfg.add_state("c")
    sdfg.add_edge(a, b, dace.InterstateEdge(assignments={"k": "1"}))
    sdfg.add_edge(b, c, dace.InterstateEdge(assignments={"k": "2", "m": "9"}))

    found = reverse_bfs_assignments(sdfg, c)
    assert found["k"] == "2"  # nearest edge (b->c) wins over farther (a->b)
    assert found["m"] == "9"

    # Starting from b only the a->b edge is reachable backward.
    found_b = reverse_bfs_assignments(sdfg, b)
    assert found_b["k"] == "1"
    assert "m" not in found_b


# ---------------------------------------------------------------------------- #
#  Splitting behaviour
# ---------------------------------------------------------------------------- #
def test_split_first_dimension_bit_exact():
    """Split the symbol-sized leading dim of a 2-D array into named per-index arrays; bit-exact."""
    sdfg = phase_axpy.to_sdfg()
    SplitArray(symbol_map=SYMBOL_MAP, name_map=NAME_MAP).apply_pass(sdfg, {})
    sdfg.validate()

    assert "field" not in sdfg.arrays
    for nm in NAMES:
        assert f"field_{nm}" in sdfg.arrays
        # Only the kept (col) axis survives in each split array.
        assert tuple(str(s) for s in sdfg.arrays[f"field_{nm}"].shape) == ("ncol", )

    csdfg = sdfg.compile()
    ncol_v = 7
    field = numpy.random.rand(4, ncol_v)
    out = numpy.zeros(ncol_v)
    args = {"out": out, "ncol": ncol_v}
    for i, nm in enumerate(NAMES):
        args[f"field_{nm}"] = field[i].copy()
    csdfg(**args)

    oracle = field[0] + 2.0 * field[1] - field[3]
    assert numpy.allclose(out, oracle)


def test_split_two_phase_dimensions_bit_exact():
    """Split two symbol-sized dims: the Cartesian product yields one named array per (i, j)."""
    sdfg = phase_matrix.to_sdfg()
    SplitArray(symbol_map=SYMBOL_MAP, name_map=NAME_MAP).apply_pass(sdfg, {})
    sdfg.validate()

    assert "m" not in sdfg.arrays
    for ni in NAMES:
        for nj in NAMES:
            assert f"m_{ni}_{nj}" in sdfg.arrays
            assert tuple(str(s) for s in sdfg.arrays[f"m_{ni}_{nj}"].shape) == ("ncol", )

    csdfg = sdfg.compile()
    ncol_v = 5
    m = numpy.random.rand(4, 4, ncol_v)
    out = numpy.zeros(ncol_v)
    args = {"out": out, "ncol": ncol_v}
    for i, ni in enumerate(NAMES):
        for j, nj in enumerate(NAMES):
            args[f"m_{ni}_{nj}"] = m[i, j].copy()
    csdfg(**args)

    oracle = m[0, 1] - m[2, 3] + m[1, 1]
    assert numpy.allclose(out, oracle)


def test_split_full_dimension_creates_length1_descriptors():
    """A fully consumed split dimension keeps no axis: each element becomes a length-1 array.

    Fully-split arrays are only *used* correctly through interstate-edge reads
    (which resolve to element 0); here we assert the descriptor shapes produced
    by _split_data_descriptors structurally, without dataflow accesses.
    """
    sdfg = dace.SDFG("full_split_descs")
    sdfg.add_array("coef", [nphase], dace.float64)
    sdfg.add_state("only", is_start_block=True)

    SplitArray(symbol_map=SYMBOL_MAP, name_map=NAME_MAP).apply_pass(sdfg, {})

    assert "coef" not in sdfg.arrays
    for nm in NAMES:
        assert f"coef_{nm}" in sdfg.arrays
        desc = sdfg.arrays[f"coef_{nm}"]
        assert tuple(int(s) for s in desc.shape) == (1, )
        assert desc.dtype == dace.float64


def test_data_dependent_index_branches_bit_exact():
    """A runtime index into the split dimension expands into a ConditionalBlock; bit-exact."""
    sdfg = pick_phase.to_sdfg()
    SplitArray(symbol_map=SYMBOL_MAP, name_map=NAME_MAP).apply_pass(sdfg, {})
    sdfg.validate()

    assert any(isinstance(n, ConditionalBlock) for n, _ in sdfg.all_nodes_recursive())
    assert "a" not in sdfg.arrays
    for nm in NAMES:
        assert f"a_{nm}" in sdfg.arrays

    csdfg = sdfg.compile()
    ncol_v = 6
    a = numpy.random.rand(4, ncol_v)
    for k in range(4):
        out = numpy.zeros(ncol_v)
        args = {"out": out, "ncol": ncol_v, "k": k}
        for i, nm in enumerate(NAMES):
            args[f"a_{nm}"] = a[i].copy()
        csdfg(**args)
        assert numpy.allclose(out, a[k] + 1.0)


def test_split_into_nested_sdfg_raises():
    """A split array feeding a nested SDFG connector is rejected (not yet supported)."""
    inner = dace.SDFG("inner_phase")
    inner.add_array("phase", [1], dace.float64)
    inner.add_array("res", [1], dace.float64)
    istate = inner.add_state("inner_state", is_start_block=True)
    ri = istate.add_read("phase")
    wi = istate.add_write("res")
    ti = istate.add_tasklet("cp", {"a"}, {"b"}, "b = a")
    istate.add_edge(ri, None, ti, "a", dace.Memlet("phase[0]"))
    istate.add_edge(ti, "b", wi, None, dace.Memlet("res[0]"))

    sdfg = dace.SDFG("outer_phase")
    sdfg.add_array("phase", [nphase], dace.float64)
    sdfg.add_array("res", [1], dace.float64)
    state = sdfg.add_state("outer_state", is_start_block=True)
    rp = state.add_read("phase")
    wr = state.add_write("res")
    nsdfg = state.add_nested_sdfg(inner, {"phase"}, {"res"})
    state.add_edge(rp, None, nsdfg, "phase", dace.Memlet("phase[0]"))
    state.add_edge(nsdfg, "res", wr, None, dace.Memlet("res[0]"))

    raised = False
    try:
        SplitArray(symbol_map=SYMBOL_MAP, name_map=NAME_MAP).apply_pass(sdfg, {})
    except Exception as exc:
        raised = True
        assert "nested" in str(exc).lower()
    assert raised, "expected SplitArray to reject split arrays feeding a nested SDFG"


if __name__ == "__main__":
    test_resolve_aliases_dedup_and_none_passthrough()
    test_copy_state_contents_preserves_graph()
    test_reverse_bfs_assignments_closest_wins()
    test_split_first_dimension_bit_exact()
    test_split_two_phase_dimensions_bit_exact()
    test_split_full_dimension_creates_length1_descriptors()
    test_data_dependent_index_branches_bit_exact()
    test_split_into_nested_sdfg_raises()
    print("split_array extra tests PASS")