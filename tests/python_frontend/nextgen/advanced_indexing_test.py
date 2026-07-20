# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
NumPy advanced (array-valued) indexing is not lowered by the nextgen frontend
yet, and must degrade to a callback rather than be mistaken for scalar
indirection.

The two features look alike in the AST — both are a subscript whose index
mentions a data container — but they are not the same. Scalar indirection
(``x[A_col[j]]``) reads *one element* to index with; advanced indexing
(``A[indices]``) uses a whole array and has its own broadcasting and
result-shape rules. Lowering the latter through the indirection mechanism
produces a tree that scores as a success (no callbacks at all) and yet cannot
be converted to a valid SDFG: the index array inherits the base array's subset,
so ``A[ind]`` for ``A: int32[M]``, ``ind: int32[N]`` emits ``ind[0:M]`` and
either fails validation or segfaults at runtime.

That is the same blind spot the accumulation tests guard: the callback
discrepancy check measures *whether* we fell back, never whether the dataflow
we emitted instead was correct. These tests pin the boundary between the two
features so the confusion cannot come back.
"""
import pytest

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

M = dace.symbol('M')
N = dace.symbol('N')


def _callbacks(root: tn.ScheduleTreeRoot):
    found = []

    def walk(node):
        for child in getattr(node, 'children', None) or []:
            if isinstance(child, tn.PythonCallbackNode):
                found.append(child)
            walk(child)

    walk(root)
    return found


@dace.program
def _index_by_array(A: dace.int32[M], ind: dace.int32[N], B: dace.int32[N]):
    return A[ind] + B


@dace.program
def _index_by_two_arrays(A: dace.float64[4, 3], rows: dace.int64[2, 2], columns: dace.int64[2, 2]):
    return A[rows, columns]


@dace.program
def _index_mixed_with_slice(A: dace.float64[20, 10, 30], ind: dace.int32[3]):
    return A[ind, 2:7:2, [15, 10, 1]]


@dace.program
def _scalar_indirection(x: dace.float64[20], col: dace.int32[10], out: dace.float64[10]):
    for i in dace.map[0:10]:
        out[i] = x[col[i]] + 1.0


@pytest.mark.parametrize('program', [_index_by_array, _index_by_two_arrays, _index_mixed_with_slice],
                         ids=['one_array', 'two_arrays', 'mixed_with_slice'])
def test_advanced_indexing_falls_back(program):
    """Until advanced indexing is implemented it must be reported as a gap, not
    silently lowered into an invalid tree."""
    assert _callbacks(nextgen.parse_program(program))


def test_mixed_indexing_does_not_raise_syntaxerror():
    """``A[ind, 2:7:2, ...]`` used to escape as a raw ``SyntaxError`` from
    re-parsing ``__in0[(__in1, 2:7:2, __in2)]`` -- an unparsed index tuple
    carries parentheses, and a slice is only legal in a bare subscript. A gap
    must surface as a categorized fallback, never as a hard crash."""
    nextgen.parse_program(_index_mixed_with_slice)


def test_scalar_indirection_still_lowers():
    """The boundary holds from the other side: a genuine one-element index read
    is still lowered as indirection, with no callback."""
    assert not _callbacks(nextgen.parse_program(_scalar_indirection))


if __name__ == '__main__':
    test_advanced_indexing_falls_back(_index_by_array)
    test_advanced_indexing_falls_back(_index_by_two_arrays)
    test_advanced_indexing_falls_back(_index_mixed_with_slice)
    test_mixed_indexing_does_not_raise_syntaxerror()
    test_scalar_indirection_still_lowers()
