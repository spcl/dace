# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A5 apply_assignment (GLOBAL_LAYOUT_DESIGN.md): layout trajectories applied end to end -- segment
clones, boundary conversions, exit conversion back to the logical interface -- stay bit-exact
against the fixture oracles, and the v1 refusals (Block trajectories) are loud."""
import pytest

from dace.libraries.layout import LayoutChange
from dace.libraries.layout.algebra import Block, Permute
from dace.transformation.layout.apply_assignment import (IDENTITY_LAYOUT, Layout, apply_assignment, segments_of)
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures
from tests.transformations.layout.multinest_fixtures_test import run_and_check

CM = Layout("perm10", (Permute((1, 0)), ))
ID = IDENTITY_LAYOUT


def split_program(program_name):
    program, _, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def test_segments_of_runs():
    assert segments_of([ID, ID, CM]) == [(0, 2, ID), (2, 3, CM)]
    assert segments_of([CM, CM]) == [(0, 2, CM)]


def test_global_colmajor_b_conflict2(n=40):
    """One segment, no boundary: B written and read column-major throughout, converted back once at
    exit (the program interface stays logical)."""
    sdfg, kernels = split_program("conflict2")
    applied = apply_assignment(sdfg, kernels, {"B": [CM, CM]})
    assert applied.segment_names["B"] == ["B__seg0_perm10"]
    assert applied.boundary_states == [] and applied.exit_state is not None
    sdfg.validate()
    assert len(line_graph(sdfg)) == 2  # conversions take no kernel position
    run_and_check(sdfg, "conflict2", n, seed=17)


def test_trajectory_with_paid_boundary_conflict2(n=40):
    """B stays row-major for nest 1 and flips to column-major for nest 2: one paid boundary
    conversion, no exit (the last write happened under identity)."""
    sdfg, kernels = split_program("conflict2")
    applied = apply_assignment(sdfg, kernels, {"B": [ID, CM]})
    assert applied.segment_names["B"] == ["B", "B__seg1_perm10"]
    assert len(applied.boundary_states) == 1 and applied.exit_state is None
    # The structural witness that the segment rewrite happened: kernel 2 reads the CLONE, not B --
    # without it, a dropped rewrite still passes the run (consumers keep reading the correct
    # original and the boundary conversion writes a dead transient).
    datas = {node.data for node in kernels[1].state.data_nodes()}
    assert "B__seg1_perm10" in datas and "B" not in datas
    sdfg.validate()
    run_and_check(sdfg, "conflict2", n, seed=18)


def test_parallel_conversions_share_one_boundary(n=32):
    """A and B both flip before nest 2 -> ONE boundary state with two parallel LayoutChange nodes
    (the A4 contract, driven from A5)."""
    sdfg, kernels = split_program("conflict2")
    applied = apply_assignment(sdfg, kernels, {"A": [ID, CM], "B": [ID, CM]})
    assert len(applied.boundary_states) == 1
    changes = [n_ for n_ in applied.boundary_states[0].nodes() if isinstance(n_, LayoutChange)]
    assert len(changes) == 2
    sdfg.validate()
    run_and_check(sdfg, "conflict2", n, seed=19)


def test_return_to_identity_conflict2(n=32):
    """A enters column-major for nest 1 and returns to identity for nest 2. A is READ-ONLY (conflict2
    only writes B and C), so the original A buffer is never mutated: the return to identity is FREE --
    nest 2 reads the untouched original, with only the ONE entry conversion (the column-major clone)
    and no restore transpose."""
    sdfg, kernels = split_program("conflict2")
    applied = apply_assignment(sdfg, kernels, {"A": [CM, ID]})
    assert applied.segment_names["A"] == ["A__seg0_perm10", "A"]
    assert len(applied.boundary_states) == 1  # only the clone entry; read-only return-to-identity is free
    datas0 = {node.data for node in kernels[0].state.data_nodes()}
    assert "A__seg0_perm10" in datas0 and "A" not in datas0  # kernel 1 was rewritten onto the clone
    datas1 = {node.data for node in kernels[1].state.data_nodes()}
    assert "A" in datas1  # kernel 2 reads the untouched original A directly (no restore)
    sdfg.validate()
    run_and_check(sdfg, "conflict2", n, seed=20)


def test_conflict3_trajectory(n=32):
    """The 3-nest program: B row-major for its producer, column-major for both transposed readers
    -- the trajectory the DP should find. One boundary, no exit."""
    sdfg, kernels = split_program("conflict3")
    applied = apply_assignment(sdfg, kernels, {"B": [ID, CM, CM]})
    assert len(applied.boundary_states) == 1 and applied.exit_state is None
    for kernel in kernels[1:]:  # both readers were rewritten onto the clone
        datas = {node.data for node in kernel.state.data_nodes()}
        assert "B__seg1_perm10" in datas and "B" not in datas
    sdfg.validate()
    assert len(line_graph(sdfg)) == 3
    run_and_check(sdfg, "conflict3", n, seed=21)


def test_identity_assignment_is_a_noop(n=24):
    sdfg, kernels = split_program("agree2")
    states_before = sdfg.number_of_nodes()
    applied = apply_assignment(sdfg, kernels, {"B": [ID, ID]})
    assert sdfg.number_of_nodes() == states_before
    assert applied.boundary_states == [] and applied.exit_state is None
    run_and_check(sdfg, "agree2", n, seed=22)


def test_block_trajectory_refused():
    sdfg, kernels = split_program("conflict2")
    blocked = Layout("block8", (Block(0, 8), ))
    with pytest.raises(NotImplementedError, match="Permute trajectories only"):
        apply_assignment(sdfg, kernels, {"B": [blocked, blocked]})


def test_length_mismatch_refused():
    sdfg, kernels = split_program("conflict2")
    with pytest.raises(ValueError, match="entries for"):
        apply_assignment(sdfg, kernels, {"B": [CM]})


if __name__ == "__main__":
    test_segments_of_runs()
    test_global_colmajor_b_conflict2()
    test_trajectory_with_paid_boundary_conflict2()
    test_parallel_conversions_share_one_boundary()
    test_return_to_identity_conflict2()
    test_conflict3_trajectory()
    test_identity_assignment_is_a_noop()
    test_block_trajectory_refused()
    test_length_mismatch_refused()
    print("apply_assignment tests PASS")
