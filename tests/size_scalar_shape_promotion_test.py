# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""A size computed in the program can be used as an array shape.

``nt = Nt + 1; np.empty(nt)`` needs ``nt`` as a symbol, but it is a data descriptor. The size is
read into a ``__sym_`` symbol on an interstate edge and substituted into the shape, leaving the
descriptor in place so it can still be read or reassigned. Each shape captures its own symbol, so
two arrays sized from the same reused name keep their own extents.
"""
import numpy as np
import pytest

import dace

N = dace.symbol('N')


@dace.program
def size_from_empty(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    b = np.empty(Nt + 1, dace.float64)
    for i in range(N):
        b[i] = a[i] * 2.0
    for i in range(N):
        out[i] = b[i]


@dace.program
def size_read_after_use(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    m = Nt + 1
    b = np.empty(m, dace.float64)
    b[0] = 1.0
    for i in range(N):
        out[i] = a[i] + m  # the size descriptor must survive its use as a shape


@dace.program
def size_reassigned_after_use(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    m = Nt + 1
    b = np.empty(m, dace.float64)
    b[0] = 1.0
    m = 99
    for i in range(N):
        out[i] = a[i] + m


@dace.program
def two_arrays_from_reassigned_size(Nt: dace.int64, out: dace.float64[1]):
    m = Nt
    b = np.empty(m, dace.float64)
    for i in range(64):
        b[i] = 1.0
    m = 2  # a second array from the same name at a different value
    c = np.empty(m, dace.float64)
    c[0] = 0.0
    out[0] = np.sum(b)  # must sum all 64 of b, not be truncated to c's size


@dace.program
def size_reused_as_index(out: dace.float64[1]):
    m = 8
    a = np.empty(m, dace.float64)
    for i in range(8):
        a[i] = i * 1.0
    m = 2
    out[0] = a[m]  # the shape symbol must not be the one this index reassigns


@dace.program
def size_from_size_one_array(nt: dace.int64[1], out: dace.float64[4]):
    b = np.empty(nt, dace.float64)
    b[0] = 1.0
    out[0] = b[0]


def test_scalar_size_as_shape():
    n, nt = 5, 7
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    size_from_empty(a, np.int64(nt), out, N=n)
    assert np.allclose(out, a * 2.0)


def test_size_descriptor_survives_its_use_as_a_shape():
    """Promotion must not delete the scalar: the program still reads it afterwards."""
    n, nt = 5, 7
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    size_read_after_use(a, np.int64(nt), out, N=n)
    assert np.allclose(out, a + (nt + 1))


def test_size_can_be_reassigned_after_use_as_a_shape():
    n, nt = 5, 7
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    size_reassigned_after_use(a, np.int64(nt), out, N=n)
    assert np.allclose(out, a + 99)


def test_two_arrays_from_a_reassigned_size_keep_their_own_extents():
    """Reusing one size name for two arrays must not collapse their extents.

    A single shared symbol gave both the last value written, so ``np.sum(b)`` returned 2.0 not 64.0.
    """
    out = np.zeros(1)
    two_arrays_from_reassigned_size(np.int64(64), out)
    assert np.isclose(out[0], 64.0)


def test_a_size_reused_as_an_index_does_not_rebind_the_extent():
    """The shape's symbol must differ from the one a later index access of the same name binds.

    Sharing it re-binds the array's extent to the reassigned value (here 2), so ``a`` is too small
    and the access goes out of bounds.
    """
    out = np.zeros(1)
    size_reused_as_index(out)
    assert np.isclose(out[0], 2.0)


def test_promotion_leaves_the_descriptor_in_place():
    sdfg = size_read_after_use.to_sdfg(simplify=False)
    # The scalar read by each ``__sym_... = <scalar>`` assignment must survive as a descriptor;
    # deleting it is what broke later reads of the size.
    sources = {
        rhs
        for e in sdfg.all_interstate_edges()
        for lhs, rhs in e.data.assignments.items() if lhs.startswith('__sym_')
    }
    assert sources, 'the size scalar must be read into a symbol'
    assert all(src in sdfg.arrays for src in sources), 'the size descriptor must survive promotion'
    sdfg.validate()


def test_shape_stays_correct_through_simplify():
    """simplify() may rewrite the promotion, but the array must keep the right extent either way.

    Run once unsimplified and once simplified; both must agree with numpy.
    """
    n, nt = 6, 9
    a = np.arange(n, dtype=np.float64)
    for simplify in (False, True):
        sdfg = size_from_empty.to_sdfg(simplify=simplify)
        out = np.zeros(n)
        sdfg(a=a, Nt=np.int64(nt), out=out, N=n)
        assert np.allclose(out, a * 2.0), f'wrong result with simplify={simplify}'


def test_size_symbol_is_assigned_before_the_allocation():
    """simplify() moves the promotion onto an edge out of the allocation's dominator.

    The allocation then read the symbol undefined and sized the array at 0, corrupting the heap on the first write.
    """
    sdfg = size_from_empty.to_sdfg(simplify=True)
    sym = str(next(iter(sdfg.arrays['b'].free_symbols)))
    lines = sdfg.generate_code()[0].clean_code.splitlines()

    # an aligned heap array reads ``new (std::align_val_t(64)) double`` and frees through ``::operator delete[](b, ..)``
    alloc = next(i for i, line in enumerate(lines) if 'b = new' in line and sym in line)
    assign = next(i for i, line in enumerate(lines) if line.strip().startswith(f'{sym} = '))
    assert assign < alloc
    assert any('delete[] b' in line or 'delete[](b' in line for line in lines), 'the array is never freed'


def test_a_size_one_array_is_read_through_a_subscript():
    """A size-1 array is a valid extent, but the assignment must read ``nt[0]``, not the pointer."""
    out = np.zeros(4)
    size_from_size_one_array(np.array([4], dtype=np.int64), out)
    assert np.allclose(out, [1.0, 0.0, 0.0, 0.0])


@dace.program
def zeros_from_size(Nt: dace.int64, out: dace.float64[1]):
    b = np.zeros(Nt + 1, dace.float64)
    out[0] = np.sum(b)


@dace.program
def ones_from_size(Nt: dace.int64, out: dace.float64[1]):
    b = np.ones(Nt + 1, dace.float64)
    out[0] = np.sum(b)


@pytest.mark.parametrize('program,expected', [(zeros_from_size, 0.0), (ones_from_size, 4.0)])
def test_the_fill_constructors_accept_a_computed_size(program, expected):
    """zeros/ones/full build their transient on their own path, which also has to promote the size."""
    out = np.zeros(1)
    program(np.int64(3), out)
    assert np.isclose(out[0], expected)


@dace.program
def size_shapes_an_array_a_slice_bound_then_reads(sizes: dace.int64[N], out: dace.float64[3, N]):
    n = sizes[0]
    buf = np.empty(n, dace.float64)
    buf[:] = 1.0
    out[0, :n] = buf


@dace.program
def compound_size_shapes_an_array_a_slice_bound_then_reads(sizes: dace.int64[N], out: dace.float64[3, N]):
    lp = sizes[0]
    buf = np.empty(lp + 1, dace.float64)
    buf[:] = 1.0
    out[0, :lp + 1] = buf


def extents_of(program):
    """The extent of every ``buf`` descriptor the program builds, as strings."""
    sdfg = program.to_sdfg(simplify=False)
    return {str(desc.shape[0]) for name, desc in sdfg.arrays.items() if name.startswith('buf')}


def test_a_shape_and_a_slice_bound_from_one_assignment_share_a_symbol():
    """``buf = np.empty(n)`` then ``out[0, :n] = buf`` is one value used twice. Promoting the shape
    to its own symbol made the copy ``[__sym_n_0]`` into ``[__sym_n]`` -- extents equal by
    construction that no consumer can prove equal, so the frontend refused the store."""
    assert extents_of(size_shapes_an_array_a_slice_bound_then_reads) == {'__sym_n'}

    out = np.zeros((3, 8))
    size_shapes_an_array_a_slice_bound_then_reads(np.array([5] + [0] * 7, dtype=np.int64), out, N=8)
    assert np.allclose(out[0], [1.0] * 5 + [0.0] * 3)


def test_a_compound_shape_keeps_the_arithmetic_the_slice_bound_keeps():
    """The same two uses spelled ``lp + 1``. Evaluating the shape as dataflow materialised the sum
    into a scalar transient and promoted THAT, so the extent was one opaque ``__sym_lp_plus_1``
    against the bound's ``__sym_lp + 1`` (cp2k_grid_integrate, cloudsc)."""
    assert extents_of(compound_size_shapes_an_array_a_slice_bound_then_reads) == {'__sym_lp + 1'}

    out = np.zeros((3, 8))
    compound_size_shapes_an_array_a_slice_bound_then_reads(np.array([4] + [0] * 7, dtype=np.int64), out, N=8)
    assert np.allclose(out[0], [1.0] * 5 + [0.0] * 3)


@dace.program
def two_slices_from_one_bound_expression(A_row: dace.int64[N], A_col: dace.float64[N], A_val: dace.float64[N],
                                         out: dace.float64[1]):
    cols = A_col[A_row[0]:A_row[1]]
    vals = A_val[A_row[0]:A_row[1]]
    out[0] = np.dot(cols, vals)


def test_two_slices_from_one_bound_expression_share_their_symbols():
    """Two slices spelled with the SAME bound expression must get the same symbols.

    Each ``A_row[0]`` read mints its own scalar transient, so promoting per transient gave the two
    slices four symbols and lengths no consumer could equate -- ``vals @ x[cols]`` in spmv was then
    refused with a size mismatch. The promotion is cached under the bound's TEXT for that reason.
    """
    sdfg = two_slices_from_one_bound_expression.to_sdfg(simplify=False)
    extents = {str(desc.shape[0]) for name, desc in sdfg.arrays.items() if name.startswith(('cols', 'vals'))}
    assert len(extents) == 1, f'the two slices must share one extent, got {extents}'

    rows = np.array([1, 4] + [0] * 6, dtype=np.int64)
    col = np.arange(8, dtype=np.float64)
    val = np.arange(8, dtype=np.float64) * 2.0
    out = np.zeros(1)
    two_slices_from_one_bound_expression(rows, col, val, out, N=8)
    assert np.isclose(out[0], np.dot(col[1:4], val[1:4]))


if __name__ == '__main__':
    test_scalar_size_as_shape()
    test_size_descriptor_survives_its_use_as_a_shape()
    test_size_can_be_reassigned_after_use_as_a_shape()
    test_two_arrays_from_a_reassigned_size_keep_their_own_extents()
    test_a_size_reused_as_an_index_does_not_rebind_the_extent()
    test_promotion_leaves_the_descriptor_in_place()
    test_shape_stays_correct_through_simplify()
    test_size_symbol_is_assigned_before_the_allocation()
    test_a_size_one_array_is_read_through_a_subscript()
    test_the_fill_constructors_accept_a_computed_size(zeros_from_size, 0.0)
    test_the_fill_constructors_accept_a_computed_size(ones_from_size, 4.0)
    test_a_shape_and_a_slice_bound_from_one_assignment_share_a_symbol()
    test_a_compound_shape_keeps_the_arithmetic_the_slice_bound_keeps()
    test_two_slices_from_one_bound_expression_share_their_symbols()
