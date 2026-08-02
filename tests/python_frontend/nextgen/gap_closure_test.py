# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for programs that used to fall back to the Python interpreter
in the next-generation frontend while the classic frontend lowered them fully.

Each test names the spelling that forced the fallback and asserts that the
lowered tree carries no callback at all — the same property the parallel
schedule-tree lowering check (``parser._check_callback_discrepancy``) enforces
across the whole test suite, pinned here per gap so a regression names its own
cause instead of surfacing as a discrepancy count somewhere else.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.common import FrontendError
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import from_schedule_tree

M, N = (dace.symbol(s, dtype=dace.int64) for s in ('M', 'N'))


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def _assert_no_callbacks(tree):
    reasons = [node.reason for node in _callbacks(tree)]
    assert not reasons, f'unexpected interpreter fallbacks: {reasons}'


def test_map_index_dimension_is_a_single_iteration_range():
    """``dace.map[10]`` is an INDEX, which the classic frontend reads as the
    one-iteration range ``10:10:1`` rather than rejecting as a non-slice."""

    @dace.program
    def index_map(A: dace.float64[2, 12]):
        for i in range(2):
            for j in dace.map[10]:
                A[i, j] = 1.0

    tree = nextgen.parse_program(index_map, np.zeros((2, 12)))
    _assert_no_callbacks(tree)
    maps = [node for node in tree.preorder_traversal() if isinstance(node, tn.MapScope)]
    assert len(maps) == 1
    assert maps[0].node.map.range.ranges == [(10, 10, 1)]


def test_hoisted_shape_element_keeps_its_compile_time_value():
    """An ANF temp hoisted out of a shape tuple materializes as a scalar, and a
    container name is not a size. A temp holding a compile-time value keeps it,
    so the shape stays exactly as static as the source wrote it -- reading the
    scalar back at run time would type just as well but would leave the caller
    unable to evaluate the shape it has to allocate."""

    @dace.program
    def padded(a: dace.float64[2, 3, 4, 5]):
        b = np.zeros((a.shape[0], a.shape[1] + 2, a.shape[2] + 2, a.shape[3]))
        b[:, 1:-1, 1:-1, :] = a
        return b

    a = np.random.rand(2, 3, 4, 5)
    tree = nextgen.parse_program(padded, a)
    _assert_no_callbacks(tree)
    assert tuple(tree.containers['b'].shape) == (2, 5, 6, 5)
    assert not [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]

    expected = np.zeros((2, 5, 6, 5))
    expected[:, 1:-1, 1:-1, :] = a
    assert np.allclose(from_schedule_tree(tree)(a.copy()), expected)


def test_runtime_shape_element_is_promoted_to_a_symbol():
    """A size the program COMPUTES cannot be a compile-time value, so the temp
    holding it is promoted to a symbol defined by an injected interstate
    assignment -- which no amount of compile-time value tracking could have
    resolved."""

    @dace.program
    def sized(counts: dace.int64[4], out: dace.float64[8]):
        b = np.zeros((counts[0] + 1, ))
        for i in dace.map[0:counts[0] + 1]:
            b[i] = float(i) * 2.0
        for i in dace.map[0:counts[0] + 1]:
            out[i] = b[i]

    counts, out = np.array([4, 0, 0, 0]), np.zeros(8)
    tree = nextgen.parse_program(sized, counts, out)
    _assert_no_callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 1
    assert str(tree.containers['b'].shape[0]) == assigns[0].name

    from_schedule_tree(tree)(counts, out)
    assert np.allclose(out, [0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0])


def test_runtime_shape_written_at_a_data_dependent_position():
    """The promoted size splits the program into states, which separates the
    array's declaration from its allocation. A write whose POSITION is chosen
    at runtime lands in a later state than the allocation, and has to find the
    array there anyway."""

    @dace.program
    def scattered(counts: dace.int64[4], out: dace.float64[8]):
        b = np.zeros((counts[0] + 1, ))
        b[counts[1]] = 7.0
        out[0] = b[counts[1]]

    counts, out = np.array([4, 3, 0, 0]), np.zeros(8)
    tree = nextgen.parse_program(scattered, counts, out)
    _assert_no_callbacks(tree)
    from_schedule_tree(tree)(counts, out)
    assert out[0] == 7.0


def test_returning_a_data_dependent_shape_is_refused():
    """A size the program computes cannot cross the program boundary: the
    caller allocates the return value before the call. Refused as a hard error
    -- degrading to a callback cannot help, since the callback would still have
    to write a ``__return`` the caller could not allocate."""

    @dace.program
    def returns_computed_size(counts: dace.int64[4]):
        b = np.zeros((counts[0] + 1, ))
        b[0] = 1.0
        return b

    with pytest.raises(FrontendError, match='data-dependent-shaped return values are not supported'):
        nextgen.parse_program(returns_computed_size, np.array([3, 0, 0, 0]))


def test_runtime_arange_bound_defines_the_symbol_it_is_sized_by():
    """``numpy.arange(stop)`` over a scalar names its result's dimension after
    the container holding ``stop`` (``arange_promoted_symbol_name``). Inventing
    the name is only half of it: unless the program also DEFINES the symbol,
    the size the program itself computed becomes a required program argument."""

    @dace.program
    def counted(counts: dace.int64[4], out: dace.int64[8]):
        r = np.arange(counts[0])
        for i in dace.map[0:counts[0]]:
            out[i] = r[i]

    counts, out = np.array([5, 0, 0, 0]), np.zeros(8, np.int64)
    tree = nextgen.parse_program(counted, counts, out)
    _assert_no_callbacks(tree)
    size = str(tree.containers['r'].shape[0])
    assert size in [node.name for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    sdfg = from_schedule_tree(tree)
    assert size not in sdfg.free_symbols, 'the caller cannot supply a size the program computes'
    sdfg(counts, out)
    assert np.array_equal(out, [0, 1, 2, 3, 4, 0, 0, 0])


def test_descriptor_attribute_argument_of_a_registry_call():
    """``numpy.dot(x, w.T)``: registry implementations take container NAMES,
    so the attribute read has to become one first."""

    @dace.program
    def linear(x: dace.float64[13, 14], w: dace.float64[10, 14]):
        return np.dot(x, w.T)

    _assert_no_callbacks(nextgen.parse_program(linear))


def test_compound_operand_of_a_registry_operator():
    """``-t @ B``: canonicalization leaves the negation in operand position,
    and the registry ``@`` implementation cannot consume an expression."""

    @dace.program
    def negated_matmul(A: dace.float64[4, 4], B: dace.float64[4, 4]):
        return -A @ B

    _assert_no_callbacks(nextgen.parse_program(negated_matmul))


def test_full_reduction_of_a_trailing_singleton_array():
    """``numpy.sum(a)`` over a ``(20, 1)`` array: the reduction map is aligned
    against the operand's NumPy shape, so squeezing dropped the LEADING
    dimension instead of the size-1 one."""

    @dace.program
    def total(a: dace.float64[20, 1]):
        return np.sum(a)

    tree = nextgen.parse_program(total, np.zeros((20, 1)))
    _assert_no_callbacks(tree)


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_per_axis_reduction_into_a_keepdims_target(axis):
    """``reduced[:] = numpy.sum(data, axis=k)`` where ``reduced`` keeps the
    reduced axis as a size-1 dimension. NumPy assignment squeezes size-1
    dimensions on both sides, so the reduced ``(4,)`` result lands in a
    ``(4, 1)`` target -- the shape ONNX's ``ReduceSum`` expansion writes
    (``dace/libraries/onnx/op_implementations/reduction_ops.py``). Matching the
    two by RANK rejected it, and every ONNX reduction fell back."""
    shape = (1, 5) if axis == 0 else (4, 1)

    @dace.program
    def keepdims_sum(data: dace.float64[4, 5], reduced: dace.float64[shape[0], shape[1]]):
        reduced[:] = np.sum(data, axis=axis)

    data = np.random.rand(4, 5)
    tree = nextgen.parse_program(keepdims_sum, data, np.zeros(shape))
    _assert_no_callbacks(tree)

    reduced = np.zeros(shape)
    from_schedule_tree(tree)(data=data.copy(), reduced=reduced)
    assert np.allclose(reduced, data.sum(axis=axis, keepdims=True))


def test_per_axis_reduction_of_a_degenerate_kept_dimension():
    """The kept dimensions may themselves be size 1 (``(4, 1, 5)`` reduced over
    its last axis into a ``(4, 1)`` target). Aligning only the non-degenerate
    dimensions must not leave the map parameter of a size-1 kept dimension out
    of the write subset, which reads as a write conflict."""

    @dace.program
    def degenerate_sum(data: dace.float64[4, 1, 5], reduced: dace.float64[4, 1]):
        reduced[:] = np.sum(data, axis=2)

    data = np.random.rand(4, 1, 5)
    tree = nextgen.parse_program(degenerate_sum, data, np.zeros((4, 1)))
    _assert_no_callbacks(tree)

    reduced = np.zeros((4, 1))
    from_schedule_tree(tree)(data=data.copy(), reduced=reduced)
    assert np.allclose(reduced, data.sum(axis=2))


def test_symbolic_intrinsic_map_bound():
    """``ceiling(N / 32)`` is resolved by the symbolic parser, not by the
    program (nothing defines the name — the same spelling
    ``tests/cuda_smem_test.py`` uses): over compile-time operands the call IS a
    symbolic expression, and canonicalization hoisting it out of the header
    must not make it a callback."""

    @dace.program
    def ceil_bound(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:ceiling(N / 32)]:  # noqa: F821 -- resolved symbolically
            for j in dace.map[i * 32:min(N, (i + 1) * 32)]:
                B[j] = A[j] * 2.0

    tree = nextgen.parse_program(ceil_bound, np.zeros(144), np.zeros(144))
    _assert_no_callbacks(tree)


def test_nested_program_specializes_its_shape_symbols():
    """A parameter declared ``float64[M]`` bound to a length-``k`` slice means
    the callee's ``M`` IS ``k``, including in the shapes it allocates — without
    which the caller sees a result shaped by a symbol it never passed."""

    @dace.program
    def double_it(A: dace.float64[M]):
        B = np.ndarray((M, ), dtype=np.float64)
        for i in dace.map[0:M]:
            B[i] = A[i] * 2.0
        return B

    @dace.program
    def caller(r: dace.float64[20]):
        return np.dot(double_it(r[:8]), r[:8])

    tree = nextgen.parse_program(caller, np.zeros(20))
    _assert_no_callbacks(tree)


def test_sdfg_call_result_into_a_subscript_target():
    """``B[:] = sdfg(...)``: the SDFG writes its own return container, which is
    then copied into the target subset."""

    @dace.program
    def nested(A: dace.float64[20]):
        return A + 20

    sdfg = nested.to_sdfg()

    @dace.program
    def mainprog(A: dace.float64[30], B: dace.float64[20]):
        B[:] = sdfg(A[10:])

    tree = nextgen.parse_program(mainprog, np.zeros(30), np.zeros(20))
    _assert_no_callbacks(tree)
    assert any(isinstance(node, tn.SDFGCallNode) for node in tree.preorder_traversal())


def test_indirection_through_a_structure_member():
    """``B[i, A.indices[idx]]``: a member read reaches sympy as an attribute of
    a symbol, so the memlet parse has to be pre-empted and the access routed to
    the same tasklet indirection path the plain-array spelling takes."""
    CSR = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[N], data=dace.float32[N]),
                              name='CSRMatrix')

    @dace.program
    def csr_to_dense(A: CSR, B: dace.float32[M, N]):
        for i in dace.map[0:M]:
            for idx in dace.map[A.indptr[i]:A.indptr[i + 1]]:
                B[i, A.indices[idx]] = A.data[idx]

    _assert_no_callbacks(nextgen.parse_program(csr_to_dense))


def test_python_object_array_field_is_not_opaque():
    """A ``PythonClass`` name that only bases a member read is not itself an
    operand: the member resolves to an ordinary container."""

    class Holder:
        data: dace.float64[4]

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder):
        for i in range(4):
            holder.data[i] = holder.data[i] + 1.0

    _assert_no_callbacks(nextgen.parse_program(prog))


def test_ufunc_over_a_constant_folded_closure_expression():
    """``numpy.power(x, nord + 1)`` over closure constants: the exponent folds
    to a SYMBOL-FREE symbolic value, which is a sympy number and not the
    Python one the registry's operand typing expects
    (``ufunc._sample_operand_value`` samples it into an object-dtype NumPy
    result with no typeclass, so descriptor inference returns None)."""
    d4_bg = 0.15
    nord = 2

    @dace.program
    def damping(a: dace.float64[10]):
        a[:] = np.power(d4_bg, nord + 1)

    tree = nextgen.parse_program(damping, np.zeros(10))
    _assert_no_callbacks(tree)

    a = np.zeros(10)
    from_schedule_tree(tree)(a)
    assert np.allclose(a, 0.15**3)


def test_ufunc_keyword_form_over_a_constant_folded_closure_expression():
    """The same fold, in the keyword form that lowers through deferred
    replacement expansion rather than the elementwise mechanism: the trial
    expansion types its operands through ``operators.result_type``, which
    rejects a sympy number outright."""
    d4_bg = 0.15
    nord = 2

    @dace.program
    def damping(a: dace.float64[10]):
        dd8 = np.power(d4_bg, nord + 1, dtype=np.float64)
        a[:] = dd8

    tree = nextgen.parse_program(damping, np.zeros(10))
    _assert_no_callbacks(tree)
    assert any(isinstance(node, tn.ReplacementCallNode) for node in tree.preorder_traversal())

    a = np.zeros(10)
    from_schedule_tree(tree)(a)
    assert np.allclose(a, 0.15**3)


if __name__ == '__main__':
    test_map_index_dimension_is_a_single_iteration_range()
    test_hoisted_shape_element_keeps_its_compile_time_value()
    test_runtime_shape_element_is_promoted_to_a_symbol()
    test_runtime_shape_written_at_a_data_dependent_position()
    test_returning_a_data_dependent_shape_is_refused()
    test_runtime_arange_bound_defines_the_symbol_it_is_sized_by()
    test_descriptor_attribute_argument_of_a_registry_call()
    test_compound_operand_of_a_registry_operator()
    test_full_reduction_of_a_trailing_singleton_array()
    for _axis in (0, 1, -1):
        test_per_axis_reduction_into_a_keepdims_target(_axis)
    test_per_axis_reduction_of_a_degenerate_kept_dimension()
    test_symbolic_intrinsic_map_bound()
    test_nested_program_specializes_its_shape_symbols()
    test_sdfg_call_result_into_a_subscript_target()
    test_indirection_through_a_structure_member()
    test_python_object_array_field_is_not_opaque()
    test_ufunc_over_a_constant_folded_closure_expression()
    test_ufunc_keyword_form_over_a_constant_folded_closure_expression()
