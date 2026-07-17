# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for branch-scoped bindings and join merging in the next-generation
frontend: in-place rebinds without φ, compatible array divergence merged
through a Reference re-pointed per branch (scalars merge by value copy),
unsound joins rolled back to callbacks, and the loop-entry stability rule.
"""
import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_if_inplace_scalar_rebind_no_merge_nodes():

    @dace.program
    def inplace(A: dace.float64[N], flag: dace.int32):
        x = 0.0
        if flag > 0:
            x = 1.0
        A[0] = x

    tree = nextgen.parse_program(inplace)
    assert len(_nodes_of_type(tree, tn.IfScope)) == 1
    # In-place scalar rebinding needs no φ: no merge copies anywhere
    assert not _nodes_of_type(tree, tn.CopyNode)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_if_divergent_alias_merges_with_refsets():
    from dace import data

    @dace.program
    def alias_merge(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], flag: dace.int32):
        if flag > 0:
            x = A
        else:
            x = B
        C[:] = x

    tree = nextgen.parse_program(alias_merge)
    if_scopes = _nodes_of_type(tree, tn.IfScope)
    else_scopes = _nodes_of_type(tree, tn.ElseScope)
    assert len(if_scopes) == 1 and len(else_scopes) == 1
    # Arrays are mutable: each branch re-points a merged Reference container
    # (a pointer set, preserving aliasing for writes after the join)
    if_refsets = [child for child in if_scopes[0].children if isinstance(child, tn.RefSetNode)]
    else_refsets = [child for child in else_scopes[0].children if isinstance(child, tn.RefSetNode)]
    assert len(if_refsets) == 1 and if_refsets[0].memlet.data == 'A'
    assert len(else_refsets) == 1 and else_refsets[0].memlet.data == 'B'
    merged = if_refsets[0].target
    assert merged == else_refsets[0].target
    assert isinstance(tree.containers[merged], data.Reference)
    # The read after the join consumes the merged reference
    final_copies = [
        copy for copy in _nodes_of_type(tree, tn.CopyNode) if copy.target == 'C' and copy.memlet.data == merged
    ]
    assert len(final_copies) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_if_without_else_gets_implicit_merge_branch():

    @dace.program
    def half_rebind(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], flag: dace.int32):
        x = A
        if flag > 0:
            x = B
        C[:] = x

    tree = nextgen.parse_program(half_rebind)
    if_scopes = _nodes_of_type(tree, tn.IfScope)
    else_scopes = _nodes_of_type(tree, tn.ElseScope)
    assert len(if_scopes) == 1
    # The fall-through path gets a synthesized else branch carrying its refset
    assert len(else_scopes) == 1
    if_refsets = [child for child in if_scopes[0].children if isinstance(child, tn.RefSetNode)]
    else_refsets = [child for child in else_scopes[0].children if isinstance(child, tn.RefSetNode)]
    assert len(if_refsets) == 1 and if_refsets[0].memlet.data == 'B'
    assert len(else_refsets) == 1 and else_refsets[0].memlet.data == 'A'
    assert if_refsets[0].target == else_refsets[0].target
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_if_shape_divergent_falls_back():

    @dace.program
    def shape_divergent(A: dace.float64[N], B: dace.float64[N, N], flag: dace.int32):
        if flag > 0:
            x = A
        else:
            x = B
        A[0] = x[0]

    tree = nextgen.parse_program(shape_divergent)
    # The whole chain is rolled back: no branch scopes survive
    assert not _nodes_of_type(tree, tn.IfScope)
    assert not _nodes_of_type(tree, tn.ElseScope)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert any('shape mismatch' in callback.reason for callback in callbacks)
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_static_divergence_falls_back():

    @dace.program
    def static_divergent(A: dace.float64[3], flag: dace.int32):
        x = [1.0, 2.0]
        if flag > 0:
            x = x + [3.0]
        A[0] = x[0]

    tree = nextgen.parse_program(static_divergent)
    assert not _nodes_of_type(tree, tn.IfScope)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) >= 1
    assert any('x' in callback.reason for callback in callbacks)


def test_static_same_shape_divergence_merges():

    @dace.program
    def static_merge(A: dace.float64[2], flag: dace.int32):
        if flag > 0:
            x = [1.0, 2.0]
        else:
            x = [3.0, 4.0]
        A[0] = x[0]
        A[1] = x[1]

    tree = nextgen.parse_program(static_merge)
    if_scopes = _nodes_of_type(tree, tn.IfScope)
    else_scopes = _nodes_of_type(tree, tn.ElseScope)
    assert len(if_scopes) == 1 and len(else_scopes) == 1
    # Each branch materializes its sequence and points the merged reference at it
    if_refsets = [child for child in if_scopes[0].children if isinstance(child, tn.RefSetNode)]
    else_refsets = [child for child in else_scopes[0].children if isinstance(child, tn.RefSetNode)]
    assert len(if_refsets) == 1 and len(else_refsets) == 1
    assert if_refsets[0].memlet.data in tree.constants
    assert else_refsets[0].memlet.data in tree.constants
    assert if_refsets[0].target == else_refsets[0].target
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_static_container_mix_merges():
    import numpy as np

    @dace.program
    def mixed_merge(A: dace.float64[2], flag: dace.int32):
        x = [1.0, 2.0]
        if flag > 0:
            x = np.zeros(2)
        A[0] = x[0]

    tree = nextgen.parse_program(mixed_merge)
    # The static fall-through path materializes into a synthesized else branch
    else_scopes = _nodes_of_type(tree, tn.ElseScope)
    assert len(else_scopes) == 1
    else_refsets = [child for child in else_scopes[0].children if isinstance(child, tn.RefSetNode)]
    assert len(else_refsets) == 1
    assert else_refsets[0].memlet.data in tree.constants
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_conditional_definition_kept():

    @dace.program
    def conditional_definition(A: dace.float64[N], B: dace.float64[N], flag: dace.int32):
        if flag > 0:
            y = A
            B[:] = y

    tree = nextgen.parse_program(conditional_definition)
    assert len(_nodes_of_type(tree, tn.IfScope)) == 1
    copies = _nodes_of_type(tree, tn.CopyNode)
    assert len(copies) == 1 and copies[0].memlet.data == 'A' and copies[0].target == 'B'
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_while_body_stable_rebind_ok():

    @dace.program
    def stable_loop(A: dace.float64[N]):
        s = 0.0
        i = 0
        while i < 10:
            s = s + A[i]
            i = i + 1
        A[0] = s

    tree = nextgen.parse_program(stable_loop)
    assert len(_nodes_of_type(tree, tn.WhileScope)) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_while_body_unstable_falls_back():

    @dace.program
    def unstable_loop(A: dace.float64[N]):
        x = [1.0]
        i = 0
        while i < 3:
            x = x + [2.0]
            i = i + 1
        A[0] = x[0]

    tree = nextgen.parse_program(unstable_loop)
    assert not _nodes_of_type(tree, tn.WhileScope)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert any('loop' in callback.reason for callback in callbacks)


def test_for_alias_rebind_falls_back():

    @dace.program
    def alias_in_loop(A: dace.float64[N], B: dace.float64[N]):
        x = B
        for i in range(3):
            x = A
        B[0] = x[0]

    tree = nextgen.parse_program(alias_in_loop)
    assert not _nodes_of_type(tree, tn.ForScope)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert any('loop-carried' in callback.reason for callback in callbacks)


def test_elif_chain_merges():

    @dace.program
    def elif_chain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], flag: dace.int32):
        if flag > 0:
            x = A
        elif flag < 0:
            x = B
        else:
            x = C
        A[0] = x[0]

    tree = nextgen.parse_program(elif_chain)
    assert len(_nodes_of_type(tree, tn.IfScope)) == 1
    assert len(_nodes_of_type(tree, tn.ElifScope)) == 1
    assert len(_nodes_of_type(tree, tn.ElseScope)) == 1
    # Three merge refsets, one per branch, re-pointing the same reference
    refsets = _nodes_of_type(tree, tn.RefSetNode)
    merge_targets = {refset.target for refset in refsets if refset.memlet.data in ('A', 'B', 'C')}
    assert len(merge_targets) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


if __name__ == '__main__':
    test_if_inplace_scalar_rebind_no_merge_nodes()
    test_if_divergent_alias_merges_with_refsets()
    test_if_without_else_gets_implicit_merge_branch()
    test_if_shape_divergent_falls_back()
    test_static_divergence_falls_back()
    test_static_same_shape_divergence_merges()
    test_static_container_mix_merges()
    test_conditional_definition_kept()
    test_while_body_stable_rebind_ok()
    test_while_body_unstable_falls_back()
    test_for_alias_rebind_falls_back()
    test_elif_chain_merges()
