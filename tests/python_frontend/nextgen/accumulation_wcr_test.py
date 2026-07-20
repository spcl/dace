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
import ast
import warnings

import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.canonical import passes
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


@dace.program
def _accumulate_spelled_out(A: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] = b[0] + A[i]
    return b


@dace.program
def _accumulate_commuted(A: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] = A[i] + b[0]
    return b


@dace.program
def _accumulate_commuted_chain(A: dace.float64[N], B: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] = A[i] + B[i] + b[0]
    return b


@dace.program
def _accumulate_wrong_operand_position(A: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] = A[i] - b[0]
    return b


@dace.program
def _accumulate_needs_reassociation(A: dace.float64[N], B: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    for i in dace.map[0:N]:
        b[0] = b[0] + A[i] + B[i]
    return b


@dace.program
def _self_referential_outside_map(A: dace.float64[N]):
    b = np.zeros([1], dtype=np.float64)
    b[0] = A[0] - b[0]
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
    map carries conflict resolution.

    Restricted to the literal ``+=`` spelling on purpose: for the forms below
    (``b = b + x`` and friends) nextgen deliberately diverges, because classic
    keys its decision off the ``AugAssign`` token alone and races on every
    other spelling of the same reduction.
    """
    from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree

    nextgen_has_wcr = any(wcr is not None for wcr in _write_wcrs(nextgen.parse_program(program, *arguments)))
    classic_has_wcr = 'CR:' in as_schedule_tree(program.to_sdfg(simplify=False)).as_string()
    assert nextgen_has_wcr == classic_has_wcr


# --- Normalized accumulations: reductions not spelled with an augmented operator


@pytest.mark.parametrize('program, arguments', [
    (_accumulate_spelled_out, (np.zeros(8), )),
    (_accumulate_commuted, (np.zeros(8), )),
    (_accumulate_commuted_chain, (np.zeros(8), np.zeros(8))),
])
def test_normalized_accumulation_uses_wcr(program, arguments):
    """``b = b + x``, ``b = x + b`` and ``b = x + y + b`` are the same reduction
    as ``b += x`` and must be conflict-resolved identically."""
    wcrs = _write_wcrs(nextgen.parse_program(program, *arguments))
    assert 'lambda x, y: x + y' in wcrs


def _conflict_warnings(program, *arguments):
    """The race reports raised while lowering a program."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nextgen.parse_program(program, *arguments)
    return [str(warning.message) for warning in caught if 'write conflict' in str(warning.message)]


@pytest.mark.parametrize('program, arguments', [
    (_accumulate_wrong_operand_position, (np.zeros(8), )),
    (_accumulate_needs_reassociation, (np.zeros(8), np.zeros(8))),
])
def test_unresolvable_conflict_is_reported(program, arguments):
    """A self-referential write that has no WCR equivalent still lowers as a
    race (matching classic), but must not do so silently -- that silence is
    what made this defect class invisible to the callback discrepancy check."""
    assert not any(_write_wcrs(nextgen.parse_program(program, *arguments)))
    assert _conflict_warnings(program, *arguments)


@pytest.mark.parametrize('program, arguments', [
    (_self_referential_outside_map, (np.zeros(8), )),
    (_plain_assign_in_map, (np.zeros(8), np.zeros(8))),
    (_accumulate_spelled_out, (np.zeros(8), )),
])
def test_no_conflict_report_without_a_race(program, arguments):
    """Reports are limited to writes that actually race: outside a dataflow
    scope, without a self-reference, or once normalized into a WCR."""
    assert not _conflict_warnings(program, *arguments)


# --- The canonicalization pass in isolation


def _detect(source: str) -> ast.stmt:
    """Run only :class:`DetectAccumulations` over a statement."""
    return passes.DetectAccumulations().transform_statement(ast.parse(source).body[0])


@pytest.mark.parametrize('source', [
    'b = b + x',
    'b = x + b',
    'b = x + y + b',
    'b = x - b',
    'b = b + x + y',
    'b = max(b, x)',
    'b = b',
    'b = x + y',
])
def test_pass_never_rewrites(source):
    """The pass runs over every assignment in the program, including ones that
    can never race, so it must annotate rather than transform: a rewrite of
    ``b = x + b`` would reverse a Python sequence concatenation."""
    assert ast.unparse(_detect(source)) == source


@pytest.mark.parametrize('source, operator, side', [
    ('b = b + x', ast.Add, 'left'),
    ('b = x + b', ast.Add, 'right'),
    ('b = x + y + b', ast.Add, 'right'),
    ('b[i] = b[i] * x', ast.Mult, 'left'),
    ('b.f = b.f | x', ast.BitOr, 'left'),
    ('b = b - x', ast.Sub, 'left'),
])
def test_pass_marks_accumulations(source, operator, side):
    result = _detect(source)
    assert isinstance(result.augmented_op, operator)
    assert result.accumulator_side == side


@pytest.mark.parametrize(
    'source, hazard',
    [
        ('b = x - b', True),  # Accumulator in a non-fold position of a non-commutative op
        ('b = x / b', True),
        ('b = b + x + y', True),  # Would need re-association, which is inexact for floats
        ('b = max(b, x)', True),  # A call: pending unified registry dispatch
        ('b = b', False),  # Self-copy: no conflict
        ('b = x + y', False),  # Not self-referential at all
        ('b = b + x', False),  # Detected as an accumulation instead
    ])
def test_pass_marks_undetectable_self_reference(source, hazard):
    assert (getattr(_detect(source), 'conflict_hazard', None) is not None) == hazard


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
    test_normalized_accumulation_uses_wcr(_accumulate_spelled_out, (np.zeros(8), ))
    test_normalized_accumulation_uses_wcr(_accumulate_commuted, (np.zeros(8), ))
    test_normalized_accumulation_uses_wcr(_accumulate_commuted_chain, (np.zeros(8), np.zeros(8)))
    test_unresolvable_conflict_is_reported(_accumulate_wrong_operand_position, (np.zeros(8), ))
    test_unresolvable_conflict_is_reported(_accumulate_needs_reassociation, (np.zeros(8), np.zeros(8)))
    test_no_conflict_report_without_a_race(_self_referential_outside_map, (np.zeros(8), ))
    test_no_conflict_report_without_a_race(_plain_assign_in_map, (np.zeros(8), np.zeros(8)))
    test_no_conflict_report_without_a_race(_accumulate_spelled_out, (np.zeros(8), ))
    for _source in [
            'b = b + x', 'b = x + b', 'b = x + y + b', 'b = x - b', 'b = b + x + y', 'b = max(b, x)', 'b = b',
            'b = x + y'
    ]:
        test_pass_never_rewrites(_source)
    for _source, _operator, _side in [('b = b + x', ast.Add, 'left'), ('b = x + b', ast.Add, 'right'),
                                      ('b = x + y + b', ast.Add, 'right'), ('b[i] = b[i] * x', ast.Mult, 'left'),
                                      ('b.f = b.f | x', ast.BitOr, 'left'), ('b = b - x', ast.Sub, 'left')]:
        test_pass_marks_accumulations(_source, _operator, _side)
    for _source, _hazard in [('b = x - b', True), ('b = x / b', True), ('b = b + x + y', True), ('b = max(b, x)', True),
                             ('b = b', False), ('b = x + y', False), ('b = b + x', False)]:
        test_pass_marks_undetectable_self_reference(_source, _hazard)
