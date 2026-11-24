# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A witness chooses values for input, never for what the program computes before the loop.

CloudSC's phase table ``iphase`` is filled from constants and read by the guards of the level loop. The solver used
to treat it as input and "prove" a dependence with ``iphase[1] = 7``, a value the program never stores. A guard array
the program computes is now a premise, not a choice: the conflict must hold for every content its range allows, else
the loop is ``unknown`` and the reason names the value.
"""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import PROVEN_PREFIX, AnnotateLoopKinds
from dace.transformation.passes.fold_constant_tables import FoldConstantTables

N = dace.symbol("N")
lvl = dace.symbol("lvl")
TABLE = np.array([0, 1, 2, 1, 2], dtype=np.int32)


@dace.program
def input_flag_equals_seven(x: dace.float64[N], y: dace.float64[N], flag: dace.int32[N]):
    for k in range(N):
        if flag[k] == 7:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def computed_phase_equals_seven(x: dace.float64[N], y: dace.float64[N], src: dace.int32[N]):
    phase = np.ndarray([N], dtype=np.int32)
    for j in range(N):
        phase[j] = src[j] * 2
    for k in range(N):
        if phase[k] == 7:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def phase_from_constants_equals_seven(x: dace.float64[N], y: dace.float64[N]):
    phase = np.ndarray([3], dtype=np.int32)
    phase[0] = 0
    phase[1] = 1
    phase[2] = 2
    for k in range(N):
        if phase[k % 3] == 7:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def phase_from_constants_at_most_two(x: dace.float64[N], y: dace.float64[N]):
    phase = np.ndarray([3], dtype=np.int32)
    phase[0] = 0
    phase[1] = 1
    phase[2] = 2
    for k in range(N):
        if phase[k % 3] <= 2:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def phase_from_constants_equals_two(x: dace.float64[N], y: dace.float64[N]):
    phase = np.ndarray([3], dtype=np.int32)
    phase[0] = 0
    phase[1] = 1
    phase[2] = 2
    for k in range(N):
        if phase[k % 3] == 2:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def clamped_level_at_most_two(x: dace.float64[N], y: dace.float64[N]):
    c = np.ndarray([3], dtype=np.int32)
    for j in dace.map[0:3]:
        c[j] = min(max(lvl, 0), 2)
    for k in range(N):
        if c[k % 3] <= 2:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def selected_level_at_most_two(x: dace.float64[N], y: dace.float64[N]):
    c = np.ndarray([3], dtype=np.int32)
    for j in dace.map[0:3]:
        c[j] = 1 if lvl > 3 else 2
    for k in range(N):
        if c[k % 3] <= 2:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def clamped_level_equals_three(x: dace.float64[N], y: dace.float64[N]):
    c = np.ndarray([3], dtype=np.int32)
    for j in dace.map[0:3]:
        c[j] = min(max(lvl, 0), 2)
    for k in range(N):
        if c[k % 3] == 3:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def affine_level_above_the_parameter(x: dace.float64[N], y: dace.float64[N]):
    c = np.ndarray([3], dtype=np.int32)
    for j in dace.map[0:3]:
        c[j] = lvl + 1
    for k in range(N):
        if c[k % 3] > lvl:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


def class_line(program: dace.program, loop_variable: str = "k", fold: bool = False, simplify: bool = True) -> str:
    sdfg = program.to_sdfg(simplify=simplify)
    if fold:
        assert FoldConstantTables().apply_pass(sdfg, {}) == {"phase"}
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (loop,) = [
        r
        for r in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and r.loop_variable == loop_variable
    ]
    return loop.specialization_hint.splitlines()[0]


def test_a_value_a_flag_input_must_take_is_still_a_witness():
    line = class_line(input_flag_equals_seven)
    assert line.startswith(PROVEN_PREFIX), line
    assert "flag[1]=7" in line, line


def test_a_conflict_that_needs_a_computed_value_is_unknown_and_names_it():
    line = class_line(computed_phase_equals_seven)
    assert line.startswith("unknown -- the only conflict the solver found needs phase["), line
    assert "=7, a value computed before the loop that the solver could not show is reachable" in line, line


def test_a_computed_table_whose_range_excludes_the_value_gives_no_witness():
    line = class_line(phase_from_constants_equals_seven)
    assert not line.startswith(PROVEN_PREFIX) and "needs phase" not in line, line


def test_a_computed_table_whose_range_forces_the_guard_keeps_its_witness_without_naming_elements():
    line = class_line(phase_from_constants_at_most_two)
    assert line.startswith(PROVEN_PREFIX), line
    assert "phase[" not in line, line


@pytest.mark.parametrize(
    "program", [clamped_level_at_most_two, selected_level_at_most_two, affine_level_above_the_parameter]
)
def test_a_computed_scalar_whose_clamp_select_or_affine_range_forces_the_guard_keeps_its_witness(program):
    line = class_line(program)
    assert line.startswith(PROVEN_PREFIX), line
    assert "c[" not in line, line


def test_a_computed_scalar_whose_clamp_range_excludes_the_value_gives_no_witness():
    line = class_line(clamped_level_equals_three)
    assert not line.startswith(PROVEN_PREFIX) and "needs c[" not in line, line


def test_a_constant_table_is_read_as_it_is_so_a_value_it_lacks_gives_no_witness():
    line = class_line(phase_from_constants_equals_seven, fold=True)
    assert not line.startswith(PROVEN_PREFIX) and "needs phase" not in line, line


def test_a_constant_table_is_read_as_it_is_so_a_value_it_holds_is_a_witness():
    line = class_line(phase_from_constants_equals_two, fold=True)
    assert line.startswith(PROVEN_PREFIX), line
    assert "phase[" not in line, line


if __name__ == "__main__":
    test_a_value_a_flag_input_must_take_is_still_a_witness()
    test_a_conflict_that_needs_a_computed_value_is_unknown_and_names_it()
    test_a_computed_table_whose_range_excludes_the_value_gives_no_witness()
    test_a_computed_table_whose_range_forces_the_guard_keeps_its_witness_without_naming_elements()
    for program in (clamped_level_at_most_two, selected_level_at_most_two, affine_level_above_the_parameter):
        test_a_computed_scalar_whose_clamp_select_or_affine_range_forces_the_guard_keeps_its_witness(program)
    test_a_computed_scalar_whose_clamp_range_excludes_the_value_gives_no_witness()
    test_a_constant_table_is_read_as_it_is_so_a_value_it_lacks_gives_no_witness()
    test_a_constant_table_is_read_as_it_is_so_a_value_it_holds_is_a_witness()
