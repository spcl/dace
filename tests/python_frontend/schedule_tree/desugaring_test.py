# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

import dace
from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree import callback_reason, desugar_schedule_tree_expansions


def _desugar_statements(source: str, *, global_vars=None, known_descriptors=None):
    module = ast.parse(source)
    desugared = desugar_schedule_tree_expansions(module,
                                                 filename='<test>',
                                                 global_vars=dict(global_vars or {}),
                                                 known_descriptors=known_descriptors)
    return [astutils.unparse(statement) for statement in desugared.body]


def test_schedule_tree_desugaring_materializes_analyzable_tuple_assignment_rhs():
    statements = _desugar_statements('A, B = B, A')
    assert statements == ['__stree_tuple_tmp = (B, A)', '(A, B) = __stree_tuple_tmp']


def test_schedule_tree_desugaring_leaves_non_analyzable_destructuring_rhs_direct():
    statements = _desugar_statements('A, B = make_pair()')
    assert statements == ['(A, B) = make_pair()']


def test_schedule_tree_desugaring_preserves_short_circuit_nested_index_guard():
    statements = _desugar_statements('if flag and A[b[i]] == 0:\n    out[0] = 1')

    assert len(statements) == 1
    assert '__stree_idx' not in statements[0]
    assert 'A[b[i]]' in statements[0]


def test_schedule_tree_desugaring_rewrites_while_with_hoisted_index_to_guarded_infinite_loop():
    statements = _desugar_statements('while A[b[i]] == 0:\n    i += 1')

    assert statements == [
        'while True:\n    __stree_idx = b[i]\n    if (not (A[__stree_idx] == 0)):\n        break\n    i += 1'
    ]


def test_schedule_tree_desugaring_marks_while_else_with_hoisted_index_for_callback():
    module = ast.parse('while A[b[i]] == 0:\n    i += 1\nelse:\n    out[0] = 1')
    desugared = desugar_schedule_tree_expansions(module, filename='<test>', global_vars={})

    assert len(desugared.body) == 1
    assert callback_reason(desugared.body[0]) == 'while loop test outlining with else'


def test_schedule_tree_desugaring_canonicalizes_negative_array_index():
    statements = _desugar_statements('tmp = A[-1]', known_descriptors={'A': dace.float64[5]})

    assert statements == ['tmp = A[(5 - 1)]']


def test_schedule_tree_desugaring_canonicalizes_symbolic_negative_array_index():
    n = dace.symbol('n')
    i = dace.symbol('i', integer=True, positive=True)

    statements = _desugar_statements('tmp = A[-i]', global_vars={'i': i}, known_descriptors={'A': dace.float64[n]})

    assert statements == ['tmp = A[(n - i)]']


def test_schedule_tree_desugaring_canonicalizes_negative_tuple_index_from_known_length():
    statements = _desugar_statements('t = (a, b, c)\ntmp = t[-1]')

    assert statements == ['t = (a, b, c)', 'tmp = t[(3 - 1)]']
