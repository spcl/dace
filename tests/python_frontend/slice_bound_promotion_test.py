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

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def pad_copy(buf: dace.float64[M], x: dace.float64[N], pad: dace.int64):
    buf[pad : pad + N] = x


@dace.program
def pad_copy_4d(padded: dace.float64[2, 3, M, M], x: dace.float64[2, 3, N, N], pad: dace.int64):
    padded[:, :, pad : pad + N, pad : pad + N] = x


@dace.program
def pad_copy_derived(buf: dace.float64[M], x: dace.float64[N], p: dace.int64):
    pad = p + 1
    buf[pad : pad + N] = x


def test_expression_bound_leaves_the_extent_readable():
    """The structural half: the assignment target's extent must be ``N`` itself, not a difference
    of two opaque promotions that happens to equal it. Read off the parsed graph rather than the
    exception, so this keeps failing informatively if the shape check ever moves."""
    sdfg = pad_copy.to_sdfg(simplify=False)
    writes = [
        e.data.dst_subset or e.data.subset
        for st in sdfg.states()
        for n in st.data_nodes()
        if n.data == "buf"
        for e in st.in_edges(n)
        if e.data is not None and not e.data.is_empty()
    ]
    assert writes, "no write to buf was parsed"
    extents = {dace.symbolic.simplify(sub.num_elements()) for sub in writes}
    assert extents == {dace.symbolic.pystr_to_symbolic("N")}, extents


@pytest.mark.parametrize("program", [pad_copy, pad_copy_derived])
def test_expression_bound_copies_where_numpy_does(program):
    """``pad_copy_derived`` adds the case the promotion cache used to collapse: the bound is built
    from a LOCAL derived from the argument, so both ends of the slice go through the same name."""
    n, m, p = 5, 11, 3
    x = np.random.default_rng(0).random(n)
    lo = p if program is pad_copy else p + 1
    expected = np.zeros(m)
    expected[lo : lo + n] = x

    buf = np.zeros(m)
    program(buf=buf, x=x, **({"pad": p} if program is pad_copy else {"p": p}), N=n, M=m)
    assert np.array_equal(buf, expected)


def test_expression_bound_in_several_dimensions_at_once():
    """The shape that motivated the fix: a 4-D pad where the SAME expression bounds two axes, so a
    cached promotion is reused across dimensions."""
    n, m, p = 4, 9, 2
    x = np.random.default_rng(1).random((2, 3, n, n))
    expected = np.zeros((2, 3, m, m))
    expected[:, :, p : p + n, p : p + n] = x

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
        buf[pad : pad + N] = x

    with pytest.raises(Exception):
        float_bound.to_sdfg(simplify=False)


@dace.program
def alias_bounds_a_slice(hc: dace.complex128[M, M]):
    n_iter = N
    c = np.zeros((N, N), dtype=np.complex128)
    for i in range(n_iter):
        for j in range(n_iter):
            c[i, j] = i * 10.0 + j
    hc[:n_iter, :n_iter] = c


@dace.program
def alias_sizes_an_array(res: dace.float64[N]):
    k = N
    res[:] = np.arange(k) + 1.0


@pytest.mark.parametrize("n", [1, 4])
def test_a_scalar_copied_from_a_symbol_bounds_a_slice_of_that_symbols_extent(n):
    """``n_iter = N`` then ``hc[:n_iter, :n_iter] = np.zeros((N, N))`` is how a generated kernel keeps
    a loop-carried size, and the store was refused as ``[N, N]`` into ``[__sym_n_iter, ...]``."""
    m = 7
    hc = np.zeros((m, m), dtype=np.complex128)
    alias_bounds_a_slice(hc=hc, N=n, M=m)
    expected = np.zeros((m, m), dtype=np.complex128)
    expected[:n, :n] = np.arange(n)[:, None] * 10.0 + np.arange(n)[None, :]
    assert np.array_equal(hc, expected), hc


def test_a_scalar_copied_from_a_symbol_sizes_an_array_of_that_symbols_extent():
    n = 6
    res = np.zeros(n)
    alias_sizes_an_array(res=res, N=n)
    assert np.array_equal(res, np.arange(n) + 1.0), res


@dace.program
def alias_bounds_a_slice_inside_a_branch(hc: dace.float64[M, M], flag: dace.bool):
    n_iter = N
    c = np.ones((N, N))
    if flag:
        hc[:n_iter, :n_iter] = c


@pytest.mark.parametrize("flag", [True, False])
def test_a_scalar_copied_from_a_symbol_bounds_a_slice_inside_a_branch(flag):
    """A branch nested in the assignment's region runs after it and at most once, so the copy still
    holds its symbol there -- the generated kernels guard their store with ``if uspp:``."""
    n, m = 3, 5
    hc = np.zeros((m, m))
    alias_bounds_a_slice_inside_a_branch(hc=hc, flag=flag, N=n, M=m)
    expected = np.zeros((m, m))
    if flag:
        expected[:n, :n] = 1.0
    assert np.array_equal(hc, expected), hc


@dace.program
def rebound_by_an_update(res: dace.float64[M]):
    k = N
    k += 1
    res[:k] = 1.0


@dace.program
def carried_through_a_loop(res: dace.float64[M]):
    n = N
    for it in range(3):
        res[:n] = res[:n] + 1.0
        n = n + 1


@dace.program
def defined_on_a_branch(res: dace.float64[M], flag: dace.bool):
    if flag:
        n = N
    else:
        n = N + 1
    res[:n] = 1.0


def reference_rebound_by_an_update(res, n):
    res[: n + 1] = 1.0


def reference_carried_through_a_loop(res, n):
    for it in range(3):
        res[: n + it] = res[: n + it] + 1.0


def reference_defined_on_a_branch(res, n, flag):
    res[: n if flag else n + 1] = 1.0


@pytest.mark.parametrize(
    "program, reference, extra",
    [
        (rebound_by_an_update, reference_rebound_by_an_update, {}),
        (carried_through_a_loop, reference_carried_through_a_loop, {}),
        (defined_on_a_branch, reference_defined_on_a_branch, {"flag": True}),
        (defined_on_a_branch, reference_defined_on_a_branch, {"flag": False}),
    ],
    ids=["update", "loop_carry", "branch_taken", "branch_not_taken"],
)
def test_a_scalar_that_may_no_longer_hold_its_symbol_keeps_its_own_value(program, reference, extra):
    """Relating ``n = N`` back to ``N`` is only sound where the assignment certainly ran and nothing
    rebound it since. An update, a later loop iteration or the other branch must read the scalar."""
    n, m = 4, 9
    res = np.zeros(m)
    program(res=res, N=n, M=m, **extra)
    expected = np.zeros(m)
    reference(expected, n, **extra)
    assert np.array_equal(res, expected), res


def test_a_temporary_sized_by_a_loop_carried_scalar_grows_with_it():
    """``res[:n] + 1.0`` under ``n = n + 1``: once the loop's uses of ``n`` read its closed form, so does the
    temporary's size. Left at ``n``, which then names the value on entry, the buffer held ``N`` elements while the
    third trip wrote ``N + 2``."""
    sdfg = carried_through_a_loop.to_sdfg(simplify=True)
    temporaries = [desc for desc in sdfg.arrays.values() if desc.transient and isinstance(desc, dace.data.Array)]
    assert temporaries
    for desc in temporaries:
        assert dace.symbolic.simplify(desc.shape[0] - dace.symbolic.pystr_to_symbolic("N + it")) == 0, desc.shape


@dace.program
def alias_bounds_an_update(hc: dace.float64[M, M]):
    n_iter = N
    c = np.ones((N, N))
    hc[:n_iter, :n_iter] += c


def test_a_scalar_copied_from_a_symbol_bounds_an_update_of_that_symbols_extent():
    n, m = 3, 5
    hc = np.full((m, m), 2.0)
    alias_bounds_an_update(hc=hc, N=n, M=m)
    expected = np.full((m, m), 2.0)
    expected[:n, :n] += 1.0
    assert np.array_equal(hc, expected), hc


@dace.program
def alias_bounds_a_store_then_a_loop_reads_it(hc: dace.float64[M, M], out: dace.float64[M]):
    n_iter = N
    hc[:n_iter, :n_iter] = np.ones((N, N))
    a = np.copy(hc[:, :n_iter])
    for it in range(2):
        a = np.copy(hc[:, :n_iter])
        out[:] = out + a[:, 0]


def test_a_loop_reading_the_copy_again_sees_the_extent_read_before_the_loop():
    """cegterg's shape: ``psi_k = np.copy(psi[:, :nbase_iter])`` before the Davidson loop and again
    inside it. Reading the copy as ``N`` outside the loop but as a fresh symbol inside it gave the one
    name two shapes, and the second binding was refused."""
    n, m = 3, 5
    hc = np.zeros((m, m))
    out = np.zeros(m)
    alias_bounds_a_store_then_a_loop_reads_it(hc=hc, out=out, N=n, M=m)
    expected_hc = np.zeros((m, m))
    expected_hc[:n, :n] = 1.0
    expected_out = 2.0 * expected_hc[:, 0]
    assert np.array_equal(hc, expected_hc), hc
    assert np.array_equal(out, expected_out), out


def test_a_store_in_a_loop_that_rebinds_the_copy_is_still_refused():
    """A later iteration stores through the rebound copy, so inside that loop it is not ``N``."""

    @dace.program
    def rebound_in_the_loop(hc: dace.float64[M, M]):
        n_iter = N
        for it in range(2):
            hc[:n_iter, :n_iter] = np.ones((N, N))
            n_iter = n_iter + 1

    with pytest.raises(IndexError, match="could not broadcast"):
        rebound_in_the_loop.to_sdfg(simplify=False)


def test_a_genuinely_different_extent_is_still_refused():

    @dace.program
    def larger_source(hc: dace.float64[M, M]):
        n_iter = N
        hc[:n_iter, :n_iter] = np.zeros((N + 1, N + 1))

    with pytest.raises(IndexError, match="could not broadcast"):
        larger_source.to_sdfg(simplify=False)


@dace.program
def strided_tap_add(out: dace.float64[M], x: dace.float64[N], p: dace.int64):
    lo = max(0, p)
    hi = min(N, p + 3)
    ol_lo = lo * 2 + 1
    ol_hi = ol_lo + (hi - lo) * 2
    out[ol_lo:ol_hi:2] = out[ol_lo:ol_hi:2] + x[lo:hi]


@dace.program
def strided_tap_add_in_nested_loops(out: dace.float64[M, 2], x: dace.float64[N, 2], p: dace.int64):
    for k in range(2):
        lo = max(0, p - k)
        hi = min(N, p + 3)
        ol_lo = lo * 2 + k
        ol_hi = ol_lo + (hi - lo) * 2
        for j in range(2):
            out[ol_lo:ol_hi:2, j] = out[ol_lo:ol_hi:2, j] + x[lo:hi, j]


def reference_strided_tap_add(out, x, p):
    lo, hi = max(0, p), min(len(x), p + 3)
    out[lo * 2 + 1 : lo * 2 + 1 + (hi - lo) * 2 : 2] += x[lo:hi]


def reference_strided_tap_add_in_nested_loops(out, x, p):
    for k in range(2):
        lo, hi = max(0, p - k), min(len(x), p + 3)
        out[lo * 2 + k : lo * 2 + k + (hi - lo) * 2 : 2, :] += x[lo:hi, :]


@pytest.mark.parametrize(
    "program, reference, shape",
    [
        (strided_tap_add, reference_strided_tap_add, ()),
        (strided_tap_add_in_nested_loops, reference_strided_tap_add_in_nested_loops, (2,)),
    ],
    ids=["flat", "nested_loops"],
)
def test_a_strided_slice_bounded_by_scalar_arithmetic_matches_its_operand(program, reference, shape):
    """A transposed convolution's tap: ``out[ol_lo:ol_hi:2] += x[lo:hi]`` with ``ol_hi = ol_lo + (hi - lo) * 2``,
    where ``lo``/``hi`` are scalars (``max``/``min`` results), so every bound is a tasklet's scalar. The slice is
    sized ``ceiling((ol_hi - ol_lo)/2)`` and the operand ``hi - lo``; the frontend refused them as a shape mismatch
    until the promoted ``ol_hi`` was proven from the promoted ``ol_lo``, ``hi`` and ``lo``."""
    n, m, p = 6, 16, 1
    x = np.arange(1.0, n * int(np.prod(shape)) + 1).reshape((n, *shape)).copy()
    out = np.zeros((m, *shape))
    program(out=out, x=x, p=p, N=n, M=m)
    expected = np.zeros((m, *shape))
    reference(expected, x, p)
    assert np.array_equal(out, expected), out


@dace.program
def strided_tap_after_a_leaf_is_rebound(out: dace.float64[M], x: dace.float64[N], p: dace.int64):
    lo = max(0, p)
    hi = min(N, p + 3)
    ol_lo = lo * 2 + 1
    ol_hi = ol_lo + (hi - lo) * 2
    hi = hi - 1
    out[ol_lo:ol_hi:2] = out[ol_lo:ol_hi:2] + x[lo:hi]


@dace.program
def floor_div_shape_over_a_scalar(x: dace.float64[M], p: dace.int64):
    n = max(1, p)
    a = np.ones(((n + 2 - 3) // 1 + 1,))
    x[:n] = a


def test_a_shape_with_floor_division_over_a_scalar_keeps_its_arithmetic():
    """squeezenet sizes a helper's output ``(n_iter + 2 * 1 - 3) // 1 + 1``: with ``//`` outside the integer
    arithmetic the shape became one opaque symbol, and the helper call could not relate it to ``n_iter``."""
    m = 6
    x = np.zeros(m)
    floor_div_shape_over_a_scalar(x=x, p=4, M=m)
    expected = np.zeros(m)
    expected[:4] = 1.0
    assert np.array_equal(x, expected), x


def test_a_rebound_leaf_does_not_prove_the_extent():
    """``hi`` changes after ``ol_hi`` read it, so the slice (3 elements) and ``x[lo:hi]`` (2) really differ, and
    NumPy refuses them; the proof must not match them up."""
    with pytest.raises(IndexError):
        strided_tap_after_a_leaf_is_rebound.to_sdfg(simplify=False)



@dace.program
def strided_tap_through_aliased_bounds(out: dace.float64[M], x: dace.float64[N], p: dace.int64, s: dace.int64):
    lo = max(0, p)
    hi = min(N, p + 3)
    ol_lo = lo * s + 1
    ol_hi = ol_lo + (hi - lo) * s
    i0 = lo + 0
    i1 = hi + 0
    out[ol_lo:ol_hi:s] = out[ol_lo:ol_hi:s] + x[i0:i1]


def test_a_strided_slice_matches_an_operand_bounded_by_aliases_of_unpromoted_scalars():
    """A transposed convolution's tap helper returns its bounds, so the operand reads ``x[i0:i1]`` with ``i0 = lo``
    and ``i1 = hi`` while ``lo`` and ``hi`` themselves are never promoted. Each unpromoted leaf stands for its own
    value in every recorded arithmetic, so ``ceiling((ol_hi - ol_lo)/s)`` still meets ``i1 - i0``, with ``s``
    a runtime stride."""
    n, m, p, stride = 6, 16, 1, 3
    x = np.arange(1.0, n + 1)
    out = np.zeros(m)
    strided_tap_through_aliased_bounds(out=out, x=x, p=p, s=stride, N=n, M=m)
    expected = np.zeros(m)
    lo, hi = max(0, p), min(n, p + 3)
    expected[lo * stride + 1 : lo * stride + 1 + (hi - lo) * stride : stride] += x[lo:hi]
    assert np.array_equal(out, expected), out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
