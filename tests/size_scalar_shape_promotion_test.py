# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""A size computed in the program can be used as an array shape.

``nt = Nt + 1; np.empty(nt)`` needs ``nt`` as a symbol, but it is a data descriptor. The size is
read into a ``__sym_`` symbol on an interstate edge and substituted into the shape, leaving the
descriptor in place so it can still be read or reassigned. Each shape captures its own symbol, so
two arrays sized from the same reused name keep their own extents.
"""
import numpy as np
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

    alloc = next(i for i, line in enumerate(lines) if 'new double' in line and sym in line)
    assign = next(i for i, line in enumerate(lines) if line.strip().startswith(f'{sym} = '))
    assert assign < alloc
    assert any('delete[] b' in line for line in lines), 'the array is never freed'


if __name__ == '__main__':
    test_scalar_size_as_shape()
    test_size_descriptor_survives_its_use_as_a_shape()
    test_size_can_be_reassigned_after_use_as_a_shape()
    test_two_arrays_from_a_reassigned_size_keep_their_own_extents()
    test_a_size_reused_as_an_index_does_not_rebind_the_extent()
    test_promotion_leaves_the_descriptor_in_place()
    test_shape_stays_correct_through_simplify()
    test_size_symbol_is_assigned_before_the_allocation()
