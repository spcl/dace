# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A4 relayout_on_boundary (GLOBAL_LAYOUT_DESIGN.md): relayout states inserted on line-graph
boundaries hold parallel LayoutChange nodes, are recognized (not kernels) by line_graph, and a
permute-there-and-back round trip through two boundaries reproduces the input bit-exactly while the
program stays correct."""
import numpy
import pytest

from dace.libraries.layout import LayoutChange
from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.line_graph import is_relayout_state, kernel_per_state, line_graph
from dace.transformation.layout.prepare import prepare_for_layout
from dace.transformation.layout.relayout_boundary import relayout_on_boundary

from tests.transformations.layout import multinest_programs as fixtures


def split_conflict2():
    program, _, _ = fixtures.PROGRAMS["conflict2"]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def test_round_trip_through_two_boundaries(n=32):
    """A -> A_cm (col-major) -> A_rt (back): the round trip is exact and the program output is
    untouched; the boundaries take no kernel position in the line graph."""
    sdfg, kernels = split_conflict2()
    assert len(kernels) == 2
    second = kernels[1].state
    relayout_on_boundary(sdfg, second, {"A": ("A_cm", [Permute((1, 0))])})
    relayout_on_boundary(sdfg, second, {"A_cm": ("A_rt", [Permute((1, 0))])}, make_transient=False)

    kernels_after = line_graph(sdfg)  # boundaries recognized, kernels unchanged
    assert len(kernels_after) == 2
    boundaries = [s for s in sdfg.states() if is_relayout_state(s)]
    assert len(boundaries) == 2

    inputs = fixtures.make_inputs(n, seed=11)
    outputs = fixtures.output_arrays("conflict2", n)
    a_rt = numpy.zeros((n, n))
    sdfg(A=inputs["A"].copy(), **outputs, A_rt=a_rt, N=n)
    assert numpy.allclose(a_rt, inputs["A"]), "permute round trip diverged"
    reference = fixtures.conflict2_oracle(inputs["A"])
    for name, ref in reference.items():
        assert numpy.allclose(outputs[name], ref)


def test_parallel_changes_share_one_state(n=16):
    sdfg, kernels = split_conflict2()
    boundary = relayout_on_boundary(sdfg, kernels[1].state, {
        "A": ("A_p", [Permute((1, 0))]),
        "B": ("B_p", [Permute((1, 0))]),
    })
    assert is_relayout_state(boundary)
    assert sum(1 for node in boundary.nodes() if isinstance(node, LayoutChange)) == 2
    assert len(line_graph(sdfg)) == 2

    inputs = fixtures.make_inputs(n, seed=13)
    outputs = fixtures.output_arrays("conflict2", n)
    sdfg(A=inputs["A"].copy(), **outputs, N=n)
    reference = fixtures.conflict2_oracle(inputs["A"])
    for name, ref in reference.items():
        assert numpy.allclose(outputs[name], ref)


def test_empty_change_set_refused():
    sdfg, kernels = split_conflict2()
    with pytest.raises(ValueError, match="empty change set"):
        relayout_on_boundary(sdfg, kernels[1].state, {})


def test_boundary_before_start_block_executes(n=16):
    """Inserting the boundary before a kernel that IS the start block must transfer start-block
    status -- ``add_state_before`` does not do it by itself, leaving an unreachable state whose
    LayoutChange never executes (the relaid array would silently stay unwritten). The fixture
    pipeline hides this (canonicalize's ``_assume_nonneg_syms`` state is the start block there), so
    the kernel-as-start-block shape is built directly -- externalized single-kernel SDFGs have it."""
    import dace

    M = dace.symbol("M")
    sdfg = dace.SDFG("start_kernel")
    sdfg.add_array("X", [M, M], dace.float64)
    sdfg.add_array("Y", [M, M], dace.float64)
    state = sdfg.add_state("k", is_start_block=True)
    me, mx = state.add_map("m", {"i": "0:M", "j": "0:M"})
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    state.add_memlet_path(state.add_access("X"), me, tasklet, dst_conn="a", memlet=dace.Memlet("X[i, j]"))
    state.add_memlet_path(tasklet, mx, state.add_access("Y"), src_conn="b", memlet=dace.Memlet("Y[i, j]"))
    sdfg.validate()

    boundary = relayout_on_boundary(sdfg, state, {"X": ("X_cm", [Permute((1, 0))])}, make_transient=False)
    assert sdfg.start_block is boundary
    sdfg.validate()
    assert len(line_graph(sdfg)) == 1  # reachable, recognized, takes no kernel position

    rng = numpy.random.default_rng(15)
    x = rng.random((n, n))
    y = numpy.zeros((n, n))
    x_cm = numpy.zeros((n, n))
    sdfg(X=x, Y=y, X_cm=x_cm, M=n)
    assert numpy.allclose(x_cm, x.T), "the boundary LayoutChange never executed"
    assert numpy.allclose(y, 2.0 * x)


if __name__ == "__main__":
    test_round_trip_through_two_boundaries()
    test_parallel_changes_share_one_state()
    test_empty_change_set_refused()
    test_boundary_before_start_block_executes()
    print("relayout_boundary tests PASS")
