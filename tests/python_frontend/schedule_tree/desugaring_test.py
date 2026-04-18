# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree import desugar_schedule_tree_expansions


def _desugar_statements(source: str):
    module = ast.parse(source)
    desugared = desugar_schedule_tree_expansions(module, filename='<test>', global_vars={})
    return [astutils.unparse(statement) for statement in desugared.body]


def test_schedule_tree_desugaring_materializes_analyzable_tuple_assignment_rhs():
    statements = _desugar_statements('A, B = B, A')
    assert statements == ['__stree_tuple_tmp = (B, A)', '(A, B) = __stree_tuple_tmp']


def test_schedule_tree_desugaring_leaves_non_analyzable_destructuring_rhs_direct():
    statements = _desugar_statements('A, B = make_pair()')
    assert statements == ['(A, B) = make_pair()']
