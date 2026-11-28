# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Genuine recurrences are classed ``sequential`` with a witness, also inside a helper that a caller hands a column.

CloudSC's helpers each take the column index ``k`` as a parameter and solve a 5x5 system in that column. ``k`` is
neither an array size nor a loop of the helper, so the witness search used to refuse every access that names it,
and the forward and backward substitutions read as ``unknown``. A parameter the caller fixes may appear in a witness
when the conflict holds for every value of it that keeps the accesses in bounds; ``a[i] = a[i - k]`` (in
``test_specialization_hints``) is the case it still may not prove, because ``k = 0`` carries nothing.

CloudSC's work arrays are the caller's transients, scratch a private copy could replace, so a conflict on one proved
nothing. A read that no write of its own iteration reaches first still needs an earlier iteration's value, which no
private copy holds: that conflict proves the order on scratch too.
"""

import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import (
    PROVEN_PREFIX,
    AnnotateLoopKinds,
    require_classes,
)

KLON = dace.symbol("KLON")
k = dace.symbol("k")


@dace.program
def forward_substitution(zqxn: dace.float64[KLON, 5], zqlhs: dace.float64[KLON, 5, 5]):
    for i in range(1, 5):
        for j in range(0, i):
            zqxn[k, i] = zqxn[k, i] - zqlhs[k, i, j] * zqxn[k, j]


@dace.program
def backward_substitution(zqxn: dace.float64[KLON, 5], zqlhs: dace.float64[KLON, 5, 5]):
    for i in range(0, 4):
        for j in range(4 - i, 5):
            zqxn[k, 3 - i] = zqxn[k, 3 - i] - zqlhs[k, 3 - i, j] * zqxn[k, j]
        zqxn[k, 3 - i] = zqxn[k, 3 - i] / zqlhs[k, 3 - i, 3 - i]


@dace.program
def lu_pivots(zqlhs: dace.float64[KLON, 5, 5]):
    for p in range(0, 4):
        for r in range(p + 1, 5):
            zqlhs[k, r, p] = zqlhs[k, r, p] / zqlhs[k, p, p]
            for c in range(p + 1, 5):
                zqlhs[k, r, c] = zqlhs[k, r, c] - zqlhs[k, r, p] * zqlhs[k, p, c]


def column_helper(program: dace.program, arrays: dict[str, tuple], scratch: bool = False) -> dace.SDFG:
    """``program`` as a nested SDFG a map over the columns calls with its column as ``k``, like CloudSC's helpers;
    ``scratch`` makes the arrays the caller's transients, copied in from arguments, as CloudSC's work arrays are."""
    inner = program.to_sdfg(simplify=True)
    outer = dace.SDFG(f"column_{inner.name}")
    init = outer.add_state("init", is_start_block=True)
    state = outer.add_state_after(init)
    entry, exit_ = state.add_map("column", {"c": "0:KLON"})
    call = state.add_nested_sdfg(inner, dict.fromkeys(arrays), dict.fromkeys(arrays), {"KLON": "KLON", "k": "c"})
    for name, shape in arrays.items():
        outer.add_array(name, shape, dace.float64, transient=scratch)
        if scratch:
            outer.add_array(f"in_{name}", shape, dace.float64)
            init.add_nedge(
                init.add_read(f"in_{name}"),
                init.add_write(name),
                dace.Memlet.from_array(f"in_{name}", outer.arrays[name]),
            )
        whole = dace.Memlet.from_array(name, outer.arrays[name])
        state.add_memlet_path(state.add_read(name), entry, call, dst_conn=name, memlet=whole)
        state.add_memlet_path(
            call, exit_, state.add_write(name), src_conn=name, memlet=dace.Memlet.from_array(name, outer.arrays[name])
        )
    outer.validate()
    return outer


def loop_classes(sdfg: dace.SDFG) -> dict[str, str]:
    AnnotateLoopKinds().apply_pass(sdfg, {})
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    return {loop.loop_variable: loop.specialization_hint.splitlines()[0] for loop in loops}


#: Opens the example a solver witness names.
EXAMPLE = "(example the solver found: "


@pytest.mark.parametrize("scratch", [False, True])
@pytest.mark.parametrize(
    "program, arrays, proven",
    [
        (forward_substitution, {"zqxn": [KLON, 5], "zqlhs": [KLON, 5, 5]}, ["i", "j"]),
        (backward_substitution, {"zqxn": [KLON, 5], "zqlhs": [KLON, 5, 5]}, ["i", "j"]),
        # Only the pivot loop: the row and column loops of one elimination step are independent
        (lu_pivots, {"zqlhs": [KLON, 5, 5]}, ["p"]),
    ],
)
def test_a_solve_in_the_column_a_caller_passes_is_a_proven_recurrence(program, arrays, proven, scratch):
    classes = loop_classes(column_helper(program, arrays, scratch))
    witnessed = [var for var, line in classes.items() if line.startswith(PROVEN_PREFIX) and EXAMPLE in line]
    assert witnessed == proven, classes
    assert not [line for line in classes.values() if line.startswith("unknown")], classes


def test_the_column_a_caller_passes_is_not_part_of_the_example():
    """The conflict holds in every column, so the witness names no column value."""
    classes = loop_classes(column_helper(forward_substitution, {"zqxn": [KLON, 5], "zqlhs": [KLON, 5, 5]}))
    assert classes["i"].endswith(f"{EXAMPLE}iterations i=1 and i=2)"), classes


@dace.program
def solve_from_top(x: dace.float64[KLON, 5], lhs: dace.float64[5, 5], top: dace.int64[1]):
    first = top[0]
    for lev in range(first, KLON):
        for i in range(1, 5):
            for j in range(0, i):
                x[lev, i] = x[lev, i] - lhs[i, j] * x[lev, j]


def test_a_solve_under_levels_from_a_value_read_before_them_is_a_proven_recurrence():
    """CloudSC's level loop starts at ``ncldtop``, a symbol the program assigns from its input before the loop."""
    classes = loop_classes(solve_from_top.to_sdfg(simplify=True))
    assert classes["lev"].startswith("parallel -- "), classes
    assert [var for var in ("i", "j") if classes[var].startswith(PROVEN_PREFIX) and EXAMPLE in classes[var]] == [
        "i",
        "j",
    ], classes


@dace.program
def staged_rows(a: dace.float64[KLON, 5], b: dace.float64[KLON, 5, 5], t: dace.float64[5]):
    for i in range(5):
        for j in range(5):
            t[j] = a[k, j] + i
        for m in range(5):
            b[k, i, m] = t[m] * 2.0


def test_a_buffer_each_iteration_fills_before_reading_proves_no_order():
    """``t`` is the caller's scratch, filled whole before every read: a private copy per row removes the conflict."""
    classes = loop_classes(column_helper(staged_rows, {"a": [KLON, 5], "b": [KLON, 5, 5], "t": [5]}, scratch=True))
    assert not classes["i"].startswith(PROVEN_PREFIX), classes


def test_a_loop_without_a_class_is_refused_by_name():
    sdfg = forward_substitution.to_sdfg(simplify=True)
    with pytest.raises(RuntimeError, match="no class .* on for_"):
        require_classes(sdfg)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    require_classes(sdfg)


if __name__ == "__main__":
    for scratch in (False, True):
        test_a_solve_in_the_column_a_caller_passes_is_a_proven_recurrence(
            forward_substitution, {"zqxn": [KLON, 5], "zqlhs": [KLON, 5, 5]}, ["i", "j"], scratch
        )
        test_a_solve_in_the_column_a_caller_passes_is_a_proven_recurrence(
            backward_substitution, {"zqxn": [KLON, 5], "zqlhs": [KLON, 5, 5]}, ["i", "j"], scratch
        )
        test_a_solve_in_the_column_a_caller_passes_is_a_proven_recurrence(
            lu_pivots, {"zqlhs": [KLON, 5, 5]}, ["p"], scratch
        )
    test_the_column_a_caller_passes_is_not_part_of_the_example()
    test_a_solve_under_levels_from_a_value_read_before_them_is_a_proven_recurrence()
    test_a_buffer_each_iteration_fills_before_reading_proves_no_order()
    test_a_loop_without_a_class_is_refused_by_name()
