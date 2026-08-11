# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for operators the replacement registry OVERRIDES, of which ``@`` is the
one with the most surface.

Python operators are dunder methods: what ``A + B`` or ``A @ B`` means depends
on the operand types, and DaCe records that in the registry. An implementation
that is not the stock elementwise one may contract (``@``), move storage
(``A @ StorageType.CPU_Heap``), reduce, or reshape — so neither the shape nor
the dataflow may be assumed. Inference asks the registry for the result
descriptor and lowering emits a deferred replacement call whenever the
registered implementation is not marked elementwise; the criterion is that
marker, not a list of operator names.

Treating ``@`` as an ordinary binary operator produced a silently wrong
answer: ``(24, 12) @ (12, 48)`` typed as (24, 12) and lowered as an
elementwise multiply that read both operands out of bounds.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N, K, M = 24, 12, 48


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_matmul_shape_and_dataflow():

    @dace.program
    def prog(a: dace.float32[N, K], b: dace.float32[K, M], out: dace.float32[N, M]):
        out[:] = a @ b

    a, b = np.random.rand(N, K).astype(np.float32), np.random.rand(K, M).astype(np.float32)
    out = np.zeros((N, M), np.float32)
    tree = nextgen.parse_program(prog, a, b, out)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The contraction goes to the registry implementation, not to a tasklet.
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(calls) == 1 and 'MatMult' in calls[0].qualname
    assert not _nodes_of_type(tree, tn.MapScope)

    tree.as_sdfg().compile()(a=a, b=b, out=out)
    assert np.allclose(out, a @ b, rtol=1e-4)


def test_matmul_delegation_chain():
    """GEMM -> GEMV -> dot product, each step a different operand rank."""

    @dace.program
    def prog(m0: dace.float32[N, K], m1: dace.float32[K, M], v0: dace.float32[M], v1: dace.float32[N],
             result: dace.float32[1]):
        result[0] = ((m0 @ m1) @ v0) @ v1

    rng = np.random.default_rng(0)
    m0 = rng.random((N, K), dtype=np.float32)
    m1 = rng.random((K, M), dtype=np.float32)
    v0, v1 = rng.random(M, dtype=np.float32), rng.random(N, dtype=np.float32)
    result = np.zeros(1, np.float32)

    tree = nextgen.parse_program(prog, m0, m1, v0, v1, result)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    tree.as_sdfg().compile()(m0=m0, m1=m1, v0=v0, v1=v1, result=result)
    assert np.allclose(result[0], ((m0 @ m1) @ v0) @ v1, rtol=1e-3)


def test_matmul_of_a_transpose_operand():
    """An operand that is itself a data-producing attribute (``a.T``)."""

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[4, 5], out: dace.float64[3, 5]):
        out[:] = a.T @ b

    a, b = np.random.rand(4, 3), np.random.rand(4, 5)
    out = np.zeros((3, 5))
    tree = nextgen.parse_program(prog, a, b, out)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    tree.as_sdfg().compile()(a=a, b=b, out=out)
    assert np.allclose(out, a.T @ b)


def test_matmul_of_a_partial_operand():
    """A subset operand is staged into its own container: passing the base
    container instead would silently contract all of ``A``."""

    @dace.program
    def prog(a: dace.float64[2, 4, 3], b: dace.float64[3, 5], out: dace.float64[4, 5]):
        out[:] = a[1] @ b

    a, b = np.random.rand(2, 4, 3), np.random.rand(3, 5)
    out = np.zeros((4, 5))
    tree = nextgen.parse_program(prog, a, b, out)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    tree.as_sdfg().compile()(a=a, b=b, out=out)
    assert np.allclose(out, a[1] @ b)


def test_batched_matmul_with_unprovable_batch_dimensions():
    """Batch dimensions that are different symbols cannot be PROVED to match;
    the implementation warns and proceeds, so inference must too."""
    B, L, MM, KK, O, NN = (dace.symbol(name) for name in ('B', 'L', 'MM', 'KK', 'O', 'NN'))

    @dace.program
    def prog(a: dace.float64[B, MM, KK], b: dace.float64[L, O, NN]):
        return a @ b

    a, b = np.random.rand(3, 6, 4), np.random.rand(3, 4, 5)
    tree = nextgen.parse_program(prog, a, b)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(a=a, b=b, B=3, L=3, MM=6, KK=4, O=4, NN=5)
    assert np.allclose(np.asarray(result), a @ b)


def test_storage_cast_operator_is_an_override_too():
    """``A @ StorageType.X`` is the same ``@``, overridden for a different
    right-hand class. Nothing about it is matmul-specific, and the general rule
    picks it up without naming it."""

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[8]):
        b = a @ dace.StorageType.CPU_Heap
        out[:] = b + 1

    a, out = np.arange(8, dtype=np.float64), np.zeros(8)
    tree = nextgen.parse_program(prog, a, out)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(calls) == 1 and calls[0].qualname.endswith('MatMult')
    tree.as_sdfg().compile()(a=a, out=out)
    assert np.allclose(out, a + 1)


def test_elementwise_operators_keep_the_fast_path():
    """The registry also registers the ordinary elementwise operators. Those
    are marked as such and must keep lowering to maps, not to deferred calls
    -- the override rule is about semantics, not about routing everything
    through the registry."""

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8], out: dace.float64[8]):
        out[:] = -a + b * 2.0

    tree = nextgen.parse_program(prog, np.zeros(8), np.zeros(8), np.zeros(8))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert not _nodes_of_type(tree, tn.ReplacementCallNode)
    assert _nodes_of_type(tree, tn.MapScope)

    a, b, out = np.random.rand(8), np.random.rand(8), np.zeros(8)
    tree.as_sdfg().compile()(a=a, b=b, out=out)
    assert np.allclose(out, -a + b * 2.0)


def test_override_criterion_is_the_registry_marker():
    """The frontend asks the registry which operators are elementwise rather
    than carrying its own list: every stock implementation is marked, and the
    ones that are not are exactly the overrides."""
    from dace.frontend.common import op_repository as oprepo
    import dace.frontend.python.replacements  # noqa: F401 -- populates the registry

    overrides = {key for key, fn in oprepo.Replacements._oprep.items() if not oprepo.is_elementwise_operator(fn)}
    assert overrides, 'expected at least the MatMult overrides to be unmarked'
    assert all(key[2] != 'Add' for key in overrides), 'plain arithmetic must be marked elementwise'
    assert ('Array', 'Array', 'MatMult') in overrides
    assert ('Array', 'StorageType', 'MatMult') in overrides


if __name__ == '__main__':
    test_matmul_shape_and_dataflow()
    test_matmul_delegation_chain()
    test_matmul_of_a_transpose_operand()
    test_matmul_of_a_partial_operand()
    test_batched_matmul_with_unprovable_batch_dimensions()
    test_storage_cast_operator_is_an_override_too()
    test_elementwise_operators_keep_the_fast_path()
    test_override_criterion_is_the_registry_marker()
