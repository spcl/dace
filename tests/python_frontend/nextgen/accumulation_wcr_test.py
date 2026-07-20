# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Conflict resolution for accumulations lowered inside dataflow scopes.

An augmented assignment inside a map is executed concurrently by every
iteration, so its write must carry a conflict-resolution function. Getting this
wrong is a *silent* miscompilation: the tree still lowers to an SDFG with no
callbacks, and the program simply returns a wrong answer, so the callback
discrepancy check cannot observe it.

A race is only probabilistically observable at runtime (measured: ~7 of 8 runs
at 65536 elements, and never at small sizes), which makes an execution-only
check an unreliable gate. These tests therefore assert the *structural*
property — the presence, absence, and form of the WCR on the write memlet —
and the numerical end-to-end behaviour is covered by the accumulation entries
of ``differential_test.py::EXECUTION_CORPUS``.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _write_wcrs(root: tn.ScheduleTreeRoot):
    """The ``wcr`` of every tasklet write in a schedule tree, in emission order."""
    found = []

    def walk(node):
        for child in getattr(node, 'children', None) or []:
            if isinstance(child, tn.TaskletNode):
                found.extend(memlet.wcr for memlet in child.out_memlets.values())
            walk(child)

    walk(root)
    return found


@dace.program
def _accumulate_scalar(A: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] += A[i]
    return b


@dace.program
def _accumulate_indexed(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] += A[i]


@dace.program
def _accumulate_nested(A: dace.float64[N], B: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            b[0] += A[i] * B[j]
    return b


@dace.program
def _accumulate_outside_map(A: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    b[0] += A[0]
    return b


@dace.program
def _plain_assign_in_map(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] + 1.0


@dace.program
def _accumulate_product(A: dace.float64[N]):
    b = np.ones([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] *= A[i]
    return b


def test_accumulation_into_map_invariant_target_uses_wcr():
    """``b[0] += A[i]`` inside a map races without conflict resolution."""
    wcrs = _write_wcrs(nextgen.parse_program(_accumulate_scalar, np.zeros(8)))
    assert 'lambda x, y: x + y' in wcrs


def test_accumulation_into_indexed_target_uses_wcr():
    """Matches the classic frontend, which conflict-resolves any in-map
    accumulation unless ``frontend.avoid_wcr`` is set."""
    wcrs = _write_wcrs(nextgen.parse_program(_accumulate_indexed, np.zeros(8), np.zeros(8)))
    assert 'lambda x, y: x + y' in wcrs


def test_accumulation_under_nested_maps_uses_wcr():
    """Conflict resolution must survive more than one enclosing dataflow scope."""
    wcrs = _write_wcrs(nextgen.parse_program(_accumulate_nested, np.zeros(8), np.zeros(8)))
    assert 'lambda x, y: x + y' in wcrs


def test_accumulation_operator_selects_matching_wcr():
    """The conflict-resolution function follows the augmented operator."""
    wcrs = _write_wcrs(nextgen.parse_program(_accumulate_product, np.zeros(8)))
    assert 'lambda x, y: x * y' in wcrs


def test_accumulation_outside_dataflow_scope_has_no_wcr():
    """Outside a map there is no concurrency, so no conflict resolution."""
    wcrs = _write_wcrs(nextgen.parse_program(_accumulate_outside_map, np.zeros(8)))
    assert all(wcr is None for wcr in wcrs)


def test_plain_assignment_in_map_has_no_wcr():
    """Only *augmented* assignments accumulate; a plain write must not."""
    wcrs = _write_wcrs(nextgen.parse_program(_plain_assign_in_map, np.zeros(8), np.zeros(8)))
    assert all(wcr is None for wcr in wcrs)


@pytest.mark.parametrize('program, arguments', [
    (_accumulate_scalar, (np.zeros(8), )),
    (_accumulate_indexed, (np.zeros(8), np.zeros(8))),
    (_plain_assign_in_map, (np.zeros(8), np.zeros(8))),
])
def test_matches_classic_wcr_decision(program, arguments):
    """The nextgen and classic frontends must agree on whether a write in a
    map carries conflict resolution."""
    from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree

    nextgen_has_wcr = any(wcr is not None for wcr in _write_wcrs(nextgen.parse_program(program, *arguments)))
    classic_has_wcr = 'CR:' in as_schedule_tree(program.to_sdfg(simplify=False)).as_string()
    assert nextgen_has_wcr == classic_has_wcr


if __name__ == '__main__':
    test_accumulation_into_map_invariant_target_uses_wcr()
    test_accumulation_into_indexed_target_uses_wcr()
    test_accumulation_under_nested_maps_uses_wcr()
    test_accumulation_operator_selects_matching_wcr()
    test_accumulation_outside_dataflow_scope_has_no_wcr()
    test_plain_assignment_in_map_has_no_wcr()
    test_matches_classic_wcr_decision(_accumulate_scalar, (np.zeros(8), ))
    test_matches_classic_wcr_decision(_accumulate_indexed, (np.zeros(8), np.zeros(8)))
    test_matches_classic_wcr_decision(_plain_assign_in_map, (np.zeros(8), np.zeros(8)))
