# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A redundant copy whose extent is spelled ``floor`` in the descriptor and ``int_floor`` in the memlet still folds."""

import numpy as np

import dace
from dace.libraries.standard import Reduce
from dace.transformation.dataflow import RedundantArray, RedundantSecondArray
from dace.transformation.dataflow.redundant_array import RedundantMapReadCopy, RemoveSliceView

N = dace.symbol("N")
K = dace.symbol("K")
#: Python's ``//`` on an expression: sympy ``floor``, as a shape annotation spells it.
EXTENT = (N - 7) // 2 + 1
#: The parser's ``//``: ``int_floor``, as a memlet string spells it.
RANGE = "0:(N - 7) // 2 + 1"
#: The same extent declared through the parser: ``int_floor`` in a descriptor.
PARSED_EXTENT = "(N - 7) // 2 + 1"
#: ``N`` for which ``EXTENT`` is 8.
SIZE = 21


def copy_sdfg(name: str, transient_first: bool, transient_extent=EXTENT) -> tuple[dace.SDFG, dace.SDFGState]:
    """``A -> map(2x) -> t -> B`` when ``transient_first``, else ``A -> t -> map(2x) -> B``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [EXTENT], dace.float64)
    sdfg.add_array("B", [EXTENT], dace.float64)
    sdfg.add_transient("t", [transient_extent], dace.float64)
    state = sdfg.add_state()
    a, t, b = state.add_read("A"), state.add_access("t"), state.add_write("B")
    src, dst = (a, t) if transient_first else (t, b)
    state.add_mapped_tasklet(
        "double",
        {"i": RANGE},
        {"x": dace.Memlet(f"{src.data}[i]")},
        "y = 2 * x",
        {"y": dace.Memlet(f"{dst.data}[i]")},
        external_edges=True,
        input_nodes={src.data: src},
        output_nodes={dst.data: dst},
    )
    if transient_first:
        state.add_nedge(t, b, dace.Memlet(f"t[{RANGE}] -> [{RANGE}]"))
    else:
        state.add_nedge(a, t, dace.Memlet(f"A[{RANGE}] -> [{RANGE}]"))
    sdfg.validate()
    return sdfg, state


def run_double(sdfg: dace.SDFG, seed: int) -> None:
    a = np.random.default_rng(seed).random((SIZE - 7) // 2 + 1)
    b = np.zeros_like(a)
    sdfg(A=a, B=b, N=SIZE)
    np.testing.assert_allclose(b, 2.0 * a)


def test_redundant_array_folds_a_copy_spelled_with_int_floor():
    sdfg, state = copy_sdfg("redundant_array_floor_spelling_first", transient_first=True)
    assert sdfg.apply_transformations(RedundantArray) == 1
    sdfg.validate()
    assert not any(n.data == "t" for n in state.data_nodes())


def test_redundant_second_array_folds_a_copy_spelled_with_int_floor():
    sdfg, state = copy_sdfg("redundant_array_floor_spelling_second", transient_first=False)
    assert sdfg.apply_transformations(RedundantSecondArray) == 1
    sdfg.validate()
    assert not any(n.data == "t" for n in state.data_nodes())


# Reproducer: Extents compared by value at every view and transient removal site; counterexamples fixed;... (cc45541710)
def test_a_copy_between_shapes_spelled_two_ways_is_no_reshape():
    """``t`` declared ``int_floor``, ``B`` declared ``floor``: one shape, so ``t`` is removed, not turned into a View."""
    sdfg, state = copy_sdfg("redundant_array_floor_spelling_no_reshape", True, transient_extent=PARSED_EXTENT)
    assert sdfg.apply_transformations(RedundantArray) == 1
    sdfg.validate()
    assert not any(n.data == "t" for n in state.data_nodes())
    run_double(sdfg, 0)


def view_into_array_sdfg(name: str, view_first: bool) -> dace.SDFG:
    """``map(2x) -> v -> B`` (``v`` a View of ``B``) when ``view_first``, else ``A -> v -> map(2x) -> B`` (``v`` a View
    of ``A``); the View is declared ``floor``, the array it views ``int_floor``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [PARSED_EXTENT if not view_first else EXTENT], dace.float64)
    sdfg.add_array("B", [PARSED_EXTENT if view_first else EXTENT], dace.float64)
    sdfg.add_view("v", [EXTENT], dace.float64)
    state = sdfg.add_state()
    a, v, b = state.add_read("A"), state.add_access("v"), state.add_write("B")
    src, dst = (a, v) if view_first else (v, b)
    state.add_mapped_tasklet(
        "double",
        {"i": RANGE},
        {"x": dace.Memlet(f"{src.data}[i]")},
        "y = 2 * x",
        {"y": dace.Memlet(f"{dst.data}[i]")},
        external_edges=True,
        input_nodes={src.data: src},
        output_nodes={dst.data: dst},
    )
    if view_first:
        state.add_edge(v, "views", b, None, dace.Memlet(f"B[{RANGE}]"))
    else:
        state.add_edge(a, None, v, "views", dace.Memlet(f"A[{RANGE}]"))
    sdfg.validate()
    return sdfg


def test_a_view_written_into_its_array_whose_shape_is_spelled_floor_is_removed():
    """``V(B) -> B``: RedundantArray refuses a View next to the array it views by design, RemoveSliceView removes it."""
    sdfg = view_into_array_sdfg("redundant_array_view_floor_spelling", view_first=True)
    assert sdfg.apply_transformations(RedundantArray) == 0
    assert sdfg.apply_transformations(RemoveSliceView) == 1
    sdfg.validate()
    assert not any(n.data == "v" for n in sdfg.start_block.data_nodes())
    run_double(sdfg, 1)


def test_a_view_read_from_its_array_whose_shape_is_spelled_floor_is_removed():
    """``A -> V(A)``: RedundantSecondArray refuses a View next to the array it views by design, RemoveSliceView
    removes it."""
    sdfg = view_into_array_sdfg("redundant_second_array_view_floor_spelling", view_first=False)
    assert sdfg.apply_transformations(RedundantSecondArray) == 0
    assert sdfg.apply_transformations(RemoveSliceView) == 1
    sdfg.validate()
    assert not any(n.data == "v" for n in sdfg.start_block.data_nodes())
    run_double(sdfg, 2)


# Reproducer: Extents compared by value at every view and transient removal site; counterexamples fixed;... (cc45541710)
def test_a_whole_view_spelled_floor_before_a_library_node_is_removed():
    """``v`` is all of ``A``, its extent spelled ``floor`` against ``A``'s ``int_floor``: an identity view, which a
    library node reads exactly as it read the View."""
    sdfg = dace.SDFG("remove_slice_view_floor_spelling")
    sdfg.add_array("A", [PARSED_EXTENT], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_view("v", [EXTENT], dace.float64)
    state = sdfg.add_state()
    v = state.add_access("v")
    red = Reduce("red", "lambda a, b: a + b", axes=None, identity=0)
    state.add_node(red)
    state.add_edge(state.add_read("A"), None, v, "views", dace.Memlet(f"A[{RANGE}]"))
    state.add_edge(v, None, red, "_in", dace.Memlet(f"v[{RANGE}]"))
    state.add_edge(red, "_out", state.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()
    assert sdfg.apply_transformations(RemoveSliceView) == 1
    sdfg.validate()
    assert not any(n.data == "v" for n in state.data_nodes())
    a, out = np.random.default_rng(3).random((SIZE - 7) // 2 + 1), np.zeros(1)
    sdfg(A=a, out=out, N=SIZE)
    np.testing.assert_allclose(out[0], a.sum())


# Reproducer: Extents compared by value at every view and transient removal site; counterexamples fixed;... (cc45541710)
def test_a_staging_copy_spelled_with_int_floor_is_folded():
    """``t = src[n, :]`` inside a map, ``t`` declared ``floor`` and the copy written ``int_floor``: the copy covers
    ``t`` and keeps its one dimension, so the readers take ``src``."""
    sdfg = dace.SDFG("staged_gather_floor_spelling")
    sdfg.add_array("src", [K, EXTENT], dace.float64)
    sdfg.add_array("idx", [EXTENT], dace.int64)
    sdfg.add_array("out", [K, EXTENT], dace.float64)
    sdfg.add_transient("t", [EXTENT], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("rows", {"n": "0:K"})
    inner_entry, inner_exit = state.add_map("cols", {"i": RANGE})
    t = state.add_access("t")
    state.add_memlet_path(state.add_read("src"), outer_entry, t, memlet=dace.Memlet(f"src[n, {RANGE}] -> [{RANGE}]"))
    tasklet = state.add_tasklet("gather", {"row", "k"}, {"o"}, "o = row[k]")
    state.add_memlet_path(t, inner_entry, tasklet, dst_conn="row", memlet=dace.Memlet(f"t[{RANGE}]", dynamic=True))
    state.add_memlet_path(
        state.add_read("idx"), outer_entry, inner_entry, tasklet, dst_conn="k", memlet=dace.Memlet("idx[i]")
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("out"), src_conn="o", memlet=dace.Memlet("out[n, i]")
    )
    sdfg.validate()
    assert sdfg.apply_transformations_repeated(RedundantMapReadCopy) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    extent, rows = (SIZE - 7) // 2 + 1, 3
    rng = np.random.default_rng(4)
    src, idx, out = rng.random((rows, extent)), rng.integers(0, extent, extent), np.zeros((rows, extent))
    sdfg(src=src, idx=idx, out=out, N=SIZE, K=rows)
    np.testing.assert_allclose(out, src[:, idx])


if __name__ == "__main__":
    test_redundant_array_folds_a_copy_spelled_with_int_floor()
    test_redundant_second_array_folds_a_copy_spelled_with_int_floor()
    test_a_copy_between_shapes_spelled_two_ways_is_no_reshape()
    test_redundant_array_removes_a_view_of_its_array_whose_shape_is_spelled_floor()
    test_redundant_second_array_removes_a_view_of_its_array_whose_shape_is_spelled_floor()
    test_a_whole_view_spelled_floor_before_a_library_node_is_removed()
    test_a_staging_copy_spelled_with_int_floor_is_folded()
