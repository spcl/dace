# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A slice bound that is an EXPRESSION over a runtime scalar.

``buf[pad : pad + N] = x`` is the padding copy every convolution wrapper writes. When ``pad`` is a
runtime scalar rather than a ``dace.symbol``, the frontend has to read it into a symbol before the
bound means anything -- and it used to do that by evaluating the whole expression into one scalar
and promoting THAT. The target extent then came out ``__sym_buf_slice - __sym_pad``: exactly ``N``,
but not visibly so, and the assignment was refused as a shape mismatch:

    IndexError: could not broadcast input array from shape [N] into shape [__sym_buf_slice - __sym_pad]

Promoting the LEAVES instead keeps the arithmetic, so the extent is ``N`` and the copy is accepted.
The tests below pin both halves: that the shapes now line up (parse), and that the elements land
where NumPy puts them (execution) -- a bound off by the padding would still parse.
"""
import numpy as np
import pytest

import dace

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def pad_copy(buf: dace.float64[M], x: dace.float64[N], pad: dace.int64):
    buf[pad:pad + N] = x


@dace.program
def pad_copy_4d(padded: dace.float64[2, 3, M, M], x: dace.float64[2, 3, N, N], pad: dace.int64):
    padded[:, :, pad:pad + N, pad:pad + N] = x


@dace.program
def pad_copy_derived(buf: dace.float64[M], x: dace.float64[N], p: dace.int64):
    pad = p + 1
    buf[pad:pad + N] = x


def test_expression_bound_leaves_the_extent_readable():
    """The structural half: the assignment target's extent must be ``N`` itself, not a difference
    of two opaque promotions that happens to equal it. Read off the parsed graph rather than the
    exception, so this keeps failing informatively if the shape check ever moves."""
    sdfg = pad_copy.to_sdfg(simplify=False)
    writes = [
        e.data.dst_subset or e.data.subset for st in sdfg.all_states() for n in st.data_nodes() if n.data == 'buf'
        for e in st.in_edges(n) if e.data is not None and not e.data.is_empty()
    ]
    assert writes, 'no write to buf was parsed'
    extents = {dace.symbolic.simplify(sub.num_elements()) for sub in writes}
    assert extents == {dace.symbolic.pystr_to_symbolic('N')}, extents


@pytest.mark.parametrize('program', [pad_copy, pad_copy_derived])
def test_expression_bound_copies_where_numpy_does(program):
    """``pad_copy_derived`` adds the case the promotion cache used to collapse: the bound is built
    from a LOCAL derived from the argument, so both ends of the slice go through the same name."""
    n, m, p = 5, 11, 3
    x = np.random.default_rng(0).random(n)
    lo = p if program is pad_copy else p + 1
    expected = np.zeros(m)
    expected[lo:lo + n] = x

    buf = np.zeros(m)
    program(buf=buf, x=x, **({'pad': p} if program is pad_copy else {'p': p}), N=n, M=m)
    assert np.array_equal(buf, expected)


def test_expression_bound_in_several_dimensions_at_once():
    """The shape that motivated the fix: a 4-D pad where the SAME expression bounds two axes, so a
    cached promotion is reused across dimensions."""
    n, m, p = 4, 9, 2
    x = np.random.default_rng(1).random((2, 3, n, n))
    expected = np.zeros((2, 3, m, m))
    expected[:, :, p:p + n, p:p + n] = x

    padded = np.zeros((2, 3, m, m))
    pad_copy_4d(padded=padded, x=x, pad=p, N=n, M=m)
    assert np.array_equal(padded, expected)


def test_a_compound_index_is_not_a_slice_bound():
    """An INDEX is not a bound: it has no extent, so promoting the whole expression is
    already exact and folding its leaves only mints a second symbol for a value the later
    scalar-to-symbol promotion already names. Worse, a leaf from an ENCLOSING scope -- a
    program argument read inside a map body -- has no descriptor in the body's own SDFG,
    which is where the promotion writes its interstate assignment.
    """

    @dace.program
    def index_by_outer_scalar(a: dace.float64[M, N], b: dace.float64[M], pad: dace.int64):
        for i in dace.map[0:M]:
            b[i] = a[i, pad + 1] + a[i, pad]

    m, n, p = 6, 9, 3
    a = np.random.default_rng(2).random((m, n))
    out = np.zeros(m)
    index_by_outer_scalar(a=a, b=out, pad=p, M=m, N=n)
    np.testing.assert_allclose(out, a[:, p + 1] + a[:, p])


def test_a_float_bound_is_still_refused():
    """The fold must not widen what a slice bound may be: a non-integer bound has no symbol to
    promote to, and silently accepting one would index by a truncated value."""

    @dace.program
    def float_bound(buf: dace.float64[M], x: dace.float64[N], pad: dace.float64):
        buf[pad:pad + N] = x

    with pytest.raises(Exception):
        float_bound.to_sdfg(simplify=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
